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
   PUBLIC inferences         ! routine called in tramle.F90
   PUBLIC inferences_final   ! routine called in nemogcm.F90

   INTEGER, PARAMETER ::   jps_e1u = 1    ! di[ x ] on u-grid
   INTEGER, PARAMETER ::   jps_e2v = 2    ! dj[ y ] on v-grid
   INTEGER, PARAMETER ::   jps_hu = 3     ! mixed-layer-depth on u-grid
   INTEGER, PARAMETER ::   jps_hv = 4     ! mixed-layer-depth on v-grid
   INTEGER, PARAMETER ::   jps_dbu = 5    ! di[ b ] on u-grid
   INTEGER, PARAMETER ::   jps_dbv = 6    ! dj[ b ] on v-grid
   INTEGER, PARAMETER ::   jps_inf = 6    ! total number of sendings for inferences

   INTEGER, PARAMETER ::   jpr_psu = 1   ! i-vertical buoyancy flux on u-grid
   INTEGER, PARAMETER ::   jpr_psv = 2   ! j-vertical buoyancy flux on v-grid
   INTEGER, PARAMETER ::   jpr_inf = 2    ! total number of inference receptions

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
         !          MLE-Fluxes-CNN          !
         ! -------------------------------- !
         ! sending Hu and Hv
         ssnd(ntypinf,jps_hu)%clname = 'E_OUT_0'
         ssnd(ntypinf,jps_hu)%laction = .TRUE.
         ssnd(ntypinf,jps_hu)%clgrid = 'U'

         ssnd(ntypinf,jps_hv)%clname = 'E_OUT_1'
         ssnd(ntypinf,jps_hv)%laction = .TRUE.
         ssnd(ntypinf,jps_hv)%clgrid = 'V'

         ! sending Delta_i_b and Delta_j_b
         ssnd(ntypinf,jps_dbu)%clname = 'E_OUT_2'
         ssnd(ntypinf,jps_dbu)%laction = .TRUE.
         ssnd(ntypinf,jps_dbu)%clgrid = 'U'

         ssnd(ntypinf,jps_dbv)%clname = 'E_OUT_3'
         ssnd(ntypinf,jps_dbv)%laction = .TRUE.
         ssnd(ntypinf,jps_dbv)%clgrid = 'V'

         ! sending e1u and e2v
         ssnd(ntypinf,jps_e1u)%clname = 'E_OUT_4'
         ssnd(ntypinf,jps_e1u)%laction = .TRUE.
         ssnd(ntypinf,jps_e1u)%clgrid = 'U'

         ssnd(ntypinf,jps_e2v)%clname = 'E_OUT_5'
         ssnd(ntypinf,jps_e2v)%laction = .TRUE.
         ssnd(ntypinf,jps_e2v)%clgrid = 'V'

         ! reception of wb_u and wb_v
         srcv(ntypinf,jpr_psu)%clname = 'E_IN_0'
         srcv(ntypinf,jpr_psu)%laction = .TRUE.
         srcv(ntypinf,jpr_psu)%clgrid = 'U'

         srcv(ntypinf,jpr_psv)%clname = 'E_IN_1'
         srcv(ntypinf,jpr_psv)%laction = .TRUE.
         srcv(ntypinf,jpr_psv)%clgrid = 'V'

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


   SUBROUTINE inferences( kt, Kbb, Kmm, Kaa, Hu, Hv, dbu, dbv )
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
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  Hu, Hv, dbu, dbv    ! sending buffer
      !
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
      ! Hu and Hv
      IF( ssnd(ntypinf,jps_hu)%laction .AND. ssnd(ntypinf,jps_hv)%laction ) THEN
         infsnd(jps_hu)%z3(:,:,ssnd(ntypinf,jps_hu)%nlvl) = Hu(:,:)
         infsnd(jps_hv)%z3(:,:,ssnd(ntypinf,jps_hv)%nlvl) = Hv(:,:)
      ENDIF  
      !
      ! Delta_i_b and Delta_j_b
      IF( ssnd(ntypinf,jps_dbu)%laction .AND. ssnd(ntypinf,jps_dbv)%laction ) THEN
         infsnd(jps_dbu)%z3(:,:,ssnd(ntypinf,jps_dbu)%nlvl) = dbu(:,:)
         infsnd(jps_dbv)%z3(:,:,ssnd(ntypinf,jps_dbv)%nlvl) = dbv(:,:)
      ENDIF
      !
      ! e1u and e2v
      IF( ssnd(ntypinf,jps_e1u)%laction .AND. ssnd(ntypinf,jps_e2v)%laction ) THEN
          infsnd(jps_e1u)%z3(:,:,ssnd(ntypinf,jps_e1u)%nlvl) = e1u(:,:)
          infsnd(jps_e2v)%z3(:,:,ssnd(ntypinf,jps_e2v)%nlvl) = e2v(:,:)
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
      ! wb_u and wb_v
      IF( srcv(ntypinf,jpr_psu)%laction .AND. srcv(ntypinf,jpr_psv)%laction ) THEN
         ext_psiu(:,:) = infrcv(jpr_psu)%z3(:,:,srcv(ntypinf,jpr_psu)%nlvl)
         ext_psiv(:,:) = infrcv(jpr_psv)%z3(:,:,srcv(ntypinf,jpr_psv)%nlvl)
         CALL iom_put( 'ext_psiu_mle', ext_psiu )
         CALL iom_put( 'ext_psiv_mle', ext_psiv )
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
