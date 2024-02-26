MODULE infmod
   !!======================================================================
   !!                       ***  MODULE  infmod  ***
   !! Machine Learning Inferences : manage connexion with external ML codes 
   !!======================================================================
   !! History :  4.2.1  ! 2023-09  (A. Barge)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   naminf          : machine learning models formulation namelist
   !!   inferences_init : initialization of Machine Learning based models
   !!   inferences      : ML based models
   !!   inf_snd         : send data to external trained model
   !!   inf_rcv         : receive inferences from external trained model
   !!----------------------------------------------------------------------
   USE oce             ! ocean fields
   USE dom_oce         ! ocean domain fields
   USE inffld          ! working fields for inferences models
   USE cpl_oasis3      ! OASIS3 coupling
   USE timing
   USE iom
   USE in_out_manager
   USE lib_mpp

   IMPLICIT NONE
   PRIVATE

   PUBLIC inf_alloc          ! function called in inferences_init 
   PUBLIC inf_dealloc        ! function called in inferences_final
   PUBLIC inferences_init    ! routine called in nemogcm.F90
   PUBLIC inferences         ! routine called in stpmlf.F90
   PUBLIC inferences_final   ! routine called in nemogcm.F90

   INTEGER, PARAMETER ::   jps_st = 1    ! sea temperature
   INTEGER, PARAMETER ::   jps_ss = 2    ! sea salinity
   INTEGER, PARAMETER ::   jps_mu = 3    ! u mask
   INTEGER, PARAMETER ::   jps_mv = 4    ! u mask
   INTEGER, PARAMETER ::   jps_inf = 4   ! total number of sendings for inferences

   INTEGER, PARAMETER ::   jpr_rho = 1   ! density inferences-computed
   INTEGER, PARAMETER ::   jpr_inf = 1   ! total number of inference receptions

   INTEGER, PARAMETER ::   jpinf = MAX(jps_inf,jpr_inf) ! Maximum number of exchanges

   TYPE( DYNARR ), SAVE, DIMENSION(jpinf) ::  infsnd, infrcv  ! sent/received inferences

   !
   !!-------------------------------------------------------------------------
   !!                    Namelist for the Inference Models
   !!-------------------------------------------------------------------------
   !                           !!** naminf namelist **
   !TYPE ::   FLD_INF              !: Field informations ...  
   !   CHARACTER(len = 32) ::         ! 
   !END TYPE FLD_INF
   !
   LOGICAL , PUBLIC ::   ln_inf    !: activate module for inference models
   
   !!-------------------------------------------------------------------------

CONTAINS

   INTEGER FUNCTION inf_alloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION inf_alloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) ALLOCATE( infrcv(jn)%z3(jpi,jpj,srcv(ntypinf,jn)%nlvl), STAT=ierr )
         IF( ssnd(ntypinf,jn)%laction ) ALLOCATE( infsnd(jn)%z3(jpi,jpj,ssnd(ntypinf,jn)%nlvl), STAT=ierr )
         inf_alloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION inf_alloc

   
   INTEGER FUNCTION inf_dealloc()
      !!----------------------------------------------------------------------
      !!             ***  FUNCTION inf_dealloc  ***
      !!----------------------------------------------------------------------
      INTEGER :: ierr
      INTEGER :: jn
      !!----------------------------------------------------------------------
      ierr = 0
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) DEALLOCATE( infrcv(jn)%z3, STAT=ierr )
         IF( ssnd(ntypinf,jn)%laction ) DEALLOCATE( infsnd(jn)%z3, STAT=ierr )
         inf_dealloc = MAX(ierr,0)
      END DO
      !
   END FUNCTION inf_dealloc


   SUBROUTINE inferences_init 
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences_init  ***
      !!
      !! ** Purpose :   Initialisation of the models that rely on external inferences
      !!
      !! ** Method  :   * Read naminf namelist
      !!                * create data for models
      !!----------------------------------------------------------------------
      !
      INTEGER ::   ios   ! Local Integer
      !!
      NAMELIST/naminf/  ln_inf
      !!----------------------------------------------------------------------
      !
      ! ================================ !
      !      Namelist informations       !
      ! ================================ !
      !
      READ  ( numnam_ref, naminf, IOSTAT = ios, ERR = 901)
901   IF( ios /= 0 )   CALL ctl_nam ( ios , 'naminf in reference namelist' )
      !
      READ  ( numnam_cfg, naminf, IOSTAT = ios, ERR = 902 )
902   IF( ios >  0 )   CALL ctl_nam ( ios , 'naminf in configuration namelist' )
      IF( lwm ) WRITE ( numond, naminf )
      !
      IF( lwp ) THEN                        ! control print
         WRITE(numout,*)
         WRITE(numout,*)'inferences_init : Setting inferences models'
         WRITE(numout,*)'~~~~~~~~~~~~~~~'
      END IF
      IF ( lwp .AND. ln_inf ) THEN
         WRITE(numout,*)'   Namelist naminf'
         WRITE(numout,*)'      Module used       ln_inf        = ', ln_inf
         WRITE(numout,*)'      Models available:'
         WRITE(numout,*)'         Stanley et al. (2020)        = ', 'T by default for now'
      ENDIF
      !
      IF( ln_inf .AND. .NOT. lk_oasis )   CALL ctl_stop( 'inferences_init : External inferences coupled via OASIS, but key_oasis3 disabled' )
      !
      !
      ! ======================================== !
      !     Define exchange needs for Models     !
      ! ======================================== !
      !
      ! default definitions of ssnd snd srcv
      srcv(ntypinf,:)%laction = .FALSE.  ;  srcv(ntypinf,:)%clgrid = 'T'  ;  srcv(ntypinf,:)%nsgn = 1.
      srcv(ntypinf,:)%nct = 1  ;  srcv(ntypinf,:)%nlvl = 1
      !
      ssnd(ntypinf,:)%laction = .FALSE.  ;  ssnd(ntypinf,:)%clgrid = 'T'  ;  ssnd(ntypinf,:)%nsgn = 1.
      ssnd(ntypinf,:)%nct = 1  ;  ssnd(ntypinf,:)%nlvl = 1
      
      IF( ln_inf ) THEN
      
         ! -------------------------------- !
         !      Kenigson et al. (2022)      !
         ! -------------------------------- !

         ! sending of sea surface temparature
         ssnd(ntypinf,jps_st)%clname = 'E_OUT_0'
         ssnd(ntypinf,jps_st)%laction = .TRUE.

         ! sending of sea surface salinity
         ssnd(ntypinf,jps_ss)%clname = 'E_OUT_1'
         ssnd(ntypinf,jps_ss)%laction = .TRUE.

         ! sending of u-grid and v-grid masks
         ssnd(ntypinf,jps_mu)%clname = 'E_OUT_2'
         ssnd(ntypinf,jps_mu)%laction = .TRUE.

         ssnd(ntypinf,jps_mv)%clname = 'E_OUT_3'
         ssnd(ntypinf,jps_mv)%laction = .TRUE.

         ! reception of sea surface density
         srcv(ntypinf,jpr_rho)%clname = 'E_IN_0'
         srcv(ntypinf,jpr_rho)%laction = .TRUE.

         ! ------------------------------ !
         ! ------------------------------ !

      END IF
      ! 
      ! ================================= !
      !   Define variables for coupling
      ! ================================= !
      CALL cpl_var(jpinf, jpinf, 1, ntypinf)
      !
      IF( inf_alloc() /= 0 )     CALL ctl_stop( 'STOP', 'inf_alloc : unable to allocate arrays' )
      IF( inffld_alloc() /= 0 )  CALL ctl_stop( 'STOP', 'inffld_alloc : unable to allocate arrays' ) 
      !
   END SUBROUTINE inferences_init


   SUBROUTINE inferences( kt, Kbb, Kmm, Kaa )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences  ***
      !!
      !! ** Purpose :   update the ocean data with the ML based models
      !!
      !! ** Method  :   *  
      !!                * 
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt            ! ocean time step
      INTEGER, INTENT(in) ::   Kbb, Kmm, Kaa ! ocean time level indices
      !
      INTEGER :: isec, info, jn                       ! local integer
      REAL(wp), DIMENSION(jpi,jpj,jpk)   ::  zdata    ! sending buffer
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('inferences')
      !
      isec = ( kt - nit000 ) * NINT( rn_Dt )       ! Date of exchange 
      info = OASIS_idle
      !
      ! ------  Prepare data to send ------
      !
      ! Sea Surface Temperature
      IF( ssnd(ntypinf,jps_st)%laction ) THEN
         infsnd(jps_st)%z3(:,:,1:ssnd(ntypinf,jps_st)%nlvl) = ts(:,:,1:ssnd(ntypinf,jps_st)%nlvl,jp_tem,Kmm)
      ENDIF  
      !s
      ! Sea Surface Salinity
      IF( ssnd(ntypinf,jps_ss)%laction ) THEN
         infsnd(jps_ss)%z3(:,:,1:ssnd(ntypinf,jps_ss)%nlvl) = ts(:,:,1:ssnd(ntypinf,jps_ss)%nlvl,jp_sal,Kmm)
      ENDIF
      !
      ! u-grid surface mask
      IF( ssnd(ntypinf,jps_mu)%laction ) THEN
          infsnd(jps_mu)%z3(:,:,1:ssnd(ntypinf,jps_mu)%nlvl) = umask(:,:,1:ssnd(ntypinf,jps_mu)%nlvl)
      ENDIF
      !
      ! v-grid surface mask
      IF( ssnd(ntypinf,jps_mv)%laction ) THEN
          infsnd(jps_mv)%z3(:,:,1:ssnd(ntypinf,jps_mv)%nlvl) = vmask(:,:,1:ssnd(ntypinf,jps_mv)%nlvl)
      ENDIF
      !
      ! ========================
      !   Proceed all sendings
      ! ========================
      !
      DO jn = 1, jpinf
         IF ( ssnd(ntypinf,jn)%laction ) THEN
            CALL cpl_snd( jn, isec, ntypinf, infsnd(jn)%z3, info)
         ENDIF
      END DO
      !
      ! .... some external operations ....
      !
      ! ==========================
      !   Proceed all receptions
      ! ==========================
      !
      DO jn = 1, jpinf
         IF( srcv(ntypinf,jn)%laction ) THEN
            CALL cpl_rcv( jn, isec, ntypinf, infrcv(jn)%z3, info)
         ENDIF
      END DO
      !
      ! ------ Distribute receptions  ------
      !
      ! Sea Surface density
      IF( srcv(ntypinf,jpr_rho)%laction ) THEN
         tmp_inf_2D(:,:) = infrcv(jpr_rho)%z3(:,:,1)
         CALL iom_put( 'St_rho_2D', tmp_inf_2D(:,:) )
      ENDIF
      !
      IF( ln_timing )   CALL timing_stop('inferences')
      !
   END SUBROUTINE inferences


   SUBROUTINE inferences_final
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE inferences_final  ***
      !!
      !! ** Purpose :   Free memory used for inferences modules
      !!
      !! ** Method  :   * Deallocate arrays
      !!----------------------------------------------------------------------
      !
      IF( inf_dealloc() /= 0 )     CALL ctl_stop( 'STOP', 'inf_dealloc : unable to free memory' )
      IF( inffld_dealloc() /= 0 )  CALL ctl_stop( 'STOP', 'inffld_dealloc : unable to free memory' )      
      !
   END SUBROUTINE inferences_final 
   !!=======================================================================
END MODULE infmod
