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
   USE sbc_oce         ! ocean surface fields
   USE inffld          ! working fields for inferences models
   USE cpl_oasis3      ! OASIS3 coupling
   USE timing
   USE iom
   USE in_out_manager
   USE lib_mpp
   USE lbclnk

   IMPLICIT NONE
   PRIVATE

   PUBLIC inf_alloc          ! function called in inferences_init 
   PUBLIC inf_dealloc        ! function called in inferences_final
   PUBLIC inferences_init    ! routine called in nemogcm.F90
   PUBLIC inferences         ! routine called in tramle.F90
   PUBLIC inferences_final   ! routine called in nemogcm.F90

   INTEGER, PARAMETER ::   jps_tmask = 1   ! t-grid mask
   INTEGER, PARAMETER ::   jps_gradb = 2   ! depth-averaged buoyancy gradient magnitude on t-grid
   INTEGER, PARAMETER ::   jps_fcor = 3    ! Coriolis parameter
   INTEGER, PARAMETER ::   jps_hml = 4     ! mixed-layer-depth on t-grid
   INTEGER, PARAMETER ::   jps_tau = 5     ! surface wind stress magnitude on t-grid
   INTEGER, PARAMETER ::   jps_q = 6       ! surface heat flux
   INTEGER, PARAMETER ::   jps_div = 7     ! depth-averaged horizontal divergence
   INTEGER, PARAMETER ::   jps_vort = 8    ! depth-averaged vertical vorticity
   INTEGER, PARAMETER ::   jps_strain = 9  ! depth-averaged strain magnitude
   INTEGER, PARAMETER ::   jps_inf = 9  ! total number of sendings

   INTEGER, PARAMETER ::   jpr_wb  = 1    ! depth-averaged subgrid vertical buoyancy flux on t-grid
   INTEGER, PARAMETER ::   jpr_inf = 1    ! total number of receptions

   INTEGER, PARAMETER ::   jpinf = MAX(jps_inf,jpr_inf) ! Maximum number of exchanges

   TYPE( DYNARR ), SAVE, DIMENSION(jpinf) ::  infsnd, infrcv  ! sent/received inferences

   !
   !!-------------------------------------------------------------------------
   !!                    Namelist for the Inference Models
   !!-------------------------------------------------------------------------
   !                           !!** naminf namelist **
   !TYPE ::   FLD_IN              !: Field informations ...  
   !   CHARACTER(len = 32) ::         ! 
   !END TYPE FLD_INF
   !
   LOGICAL , PUBLIC ::   ln_inf    !: activate module for inference models
   
   !!-------------------------------------------------------------------------
   !! * Substitutions
#  include "do_loop_substitute.h90"
#  include "domzgr_substitute.h90"
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
         ! sending gradb, FCOR, HML, TAU, Q, div, vort, strain
         ssnd(ntypinf,jps_gradb)%clname = 'E_OUT_0'
         ssnd(ntypinf,jps_gradb)%laction = .TRUE.

         ssnd(ntypinf,jps_fcor)%clname = 'E_OUT_1'
         ssnd(ntypinf,jps_fcor)%laction = .TRUE.

         ssnd(ntypinf,jps_hml)%clname = 'E_OUT_2'
         ssnd(ntypinf,jps_hml)%laction = .TRUE.

         ssnd(ntypinf,jps_tau)%clname = 'E_OUT_3'
         ssnd(ntypinf,jps_tau)%laction = .TRUE.

         ssnd(ntypinf,jps_q)%clname = 'E_OUT_4'
         ssnd(ntypinf,jps_q)%laction = .TRUE.

         ssnd(ntypinf,jps_div)%clname = 'E_OUT_5'
         ssnd(ntypinf,jps_div)%laction = .TRUE.

         ssnd(ntypinf,jps_vort)%clname = 'E_OUT_6'
         ssnd(ntypinf,jps_vort)%laction = .TRUE.

         ssnd(ntypinf,jps_strain)%clname = 'E_OUT_7'
         ssnd(ntypinf,jps_strain)%laction = .TRUE.

         ssnd(ntypinf,jps_tmask)%clname = 'E_OUT_8'
         ssnd(ntypinf,jps_tmask)%laction = .TRUE.

         ! reception of vertical buoyancy fluxes     
         srcv(ntypinf,jpr_wb)%clname = 'E_IN_0'
         srcv(ntypinf,jpr_wb)%laction = .TRUE.

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


   SUBROUTINE inferences( kt, Kbb, Kmm, Kaa, hmld, bz, uz, vz )
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
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  hmld, bz, uz, vz
      !
      !
      INTEGER :: isec, info, jn                          ! local integer
      REAL(wp), DIMENSION(jpi,jpj)   ::  zdat, zdatx, zdaty   ! working buffer
      !!----------------------------------------------------------------------
      !
      IF( ln_timing )   CALL timing_start('inferences')
      !
      isec = ( kt - nit000 ) * NINT( rn_Dt )       ! Date of exchange 
      info = OASIS_idle
      !
      ! ------  Prepare data to send ------
      !
      ! gradB
      CALL calc_2D_scal_gradient( bz, zdatx, zdaty )
      infsnd(jps_gradb)%z3(:,:,ssnd(ntypinf,jps_gradb)%nlvl) = SQRT( zdatx(:,:)**2 + zdaty(:,:)**2 )
      ! FCOR
      infsnd(jps_fcor)%z3(:,:,ssnd(ntypinf,jps_fcor)%nlvl) = ff_t(:,:)
      ! HML
      infsnd(jps_hml)%z3(:,:,ssnd(ntypinf,jps_hml)%nlvl) = hmld(:,:)
      ! Tau
      infsnd(jps_tau)%z3(:,:,ssnd(ntypinf,jps_tau)%nlvl) = taum(:,:)
      ! Heat Flux
      infsnd(jps_q)%z3(:,:,ssnd(ntypinf,jps_q)%nlvl) = qsr(:,:) + qns(:,:) 
      ! horizontal divergence
      CALL calc_2D_vec_hdiv( hmld, uz, vz, zdat )
      infsnd(jps_div)%z3(:,:,ssnd(ntypinf,jps_div)%nlvl) = zdat(:,:) 
      ! vorticity
      CALL calc_2D_vec_vort( uz, vz, zdat )
      infsnd(jps_vort)%z3(:,:,ssnd(ntypinf,jps_vort)%nlvl) = zdat(:,:)
      ! strain
      CALL calc_2D_strain_magnitude( uz, vz, zdat )
      infsnd(jps_strain)%z3(:,:,ssnd(ntypinf,jps_strain)%nlvl) = zdat(:,:)
      ! tmask
      infsnd(jps_tmask)%z3(:,:,1:ssnd(ntypinf,jps_tmask)%nlvl) = tmask(:,:,1:ssnd(ntypinf,jps_tmask)%nlvl) 
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
      ! wb
      ext_wb(:,:) = infrcv(jpr_wb)%z3(:,:,srcv(ntypinf,jpr_wb)%nlvl)
      !
      ! get streamfunction on correct grid points
      CALL invert_buoyancy_flux( ext_wb, zdatx, zdaty, ext_psiu, ext_psiv )  
      !
      ! output results
      CALL iom_put( 'ext_wb', ext_wb )
      CALL iom_put( 'ext_psiu_mle', ext_psiu )
      CALL iom_put( 'ext_psiv_mle', ext_psiv )
      !
      IF( ln_timing )   CALL timing_stop('inferences')
      !
   END SUBROUTINE inferences


   SUBROUTINE invert_buoyancy_flux( wb, gradbx, gradby, psiu, psiv )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE invert_buoyancy_flux  ***
      !!
      !! ** Purpose :   Compute streamfunction on u- and v- points 
      !!                from vertical buoyancy flux and buoyancy gradient
      !!
      !! ** Method  :   * Invert w'b' = psi x grad_b
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(inout) ::  wb, gradbx, gradby  ! vert. buoyncy flux and buoyancy gradient
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: psiu, psiv  ! computed streamfunction
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      INTEGER  :: jwgt              ! local storage integer
      REAL(wp) :: amp
      REAL(wp), DIMENSION(jpi,jpj) :: ztmpu, ztmpv  ! buffers
      !!----------------------------------------------------------------------
      !
      ! invert buoyancy fluxes
      CALL lbc_lnk( 'infmod', gradbx, 'T', 1.0_wp , gradby, 'T', 1.0_wp )
      DO_2D( nn_hls, nn_hls, nn_hls, nn_hls )
         amp = gradbx(ji,jj)**2 + gradby(ji,jj)**2
         IF ( amp == 0.0_wp ) amp = 1.0_wp
         ztmpu(ji,jj) = wb(ji,jj) * gradbx(ji,jj) / amp * tmask(ji,jj,1)
         ztmpv(ji,jj) = wb(ji,jj) * gradby(ji,jj) / amp * tmask(ji,jj,1)
      END_2D
      !
      ! interpolate to velocity points
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         ! u-grid
         jwgt = tmask(ji+1,jj,1) + tmask(ji,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         psiu(ji,jj) = ( ztmpu(ji+1,jj) + ztmpu(ji,jj) ) / REAL(jwgt,wp)

         ! v-grid
         jwgt = tmask(ji,jj+1,1) + tmask(ji,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         psiv(ji,jj) = ( ztmpv(ji,jj+1) + ztmpv(ji,jj) ) / REAL(jwgt,wp)
      END_2D
      !
   END SUBROUTINE invert_buoyancy_flux


   SUBROUTINE calc_2D_scal_gradient( scalar, gradx, grady )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_scal_gradient  ***
      !!
      !! ** Purpose :   Compute gradient of a 2D scalar field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  scalar        ! input scalar
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: gradx, grady  ! computed gradient
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      INTEGER  :: jwgt              ! local storage integer
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         ! grad in i-longitude
         gradx(ji,jj) = ( scalar(ji+1,jj) - scalar(ji,jj) ) / e1u(ji,jj) * umask(ji,jj,1)
         gradx(ji,jj) = gradx(ji,jj) + ( scalar(ji,jj) - scalar(ji-1,jj) ) / e1u(ji-1,jj) * umask(ji-1,jj,1)
         jwgt = umask(ji,jj,1) + umask(ji-1,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         gradx(ji,jj) = gradx(ji,jj) / REAL(jwgt,wp)
        
         ! grad in j-latitude
         grady(ji,jj) = ( scalar(ji,jj+1) - scalar(ji,jj) ) / e2v(ji,jj) * vmask(ji,jj,1)
         grady(ji,jj) = grady(ji,jj) + ( scalar(ji,jj) - scalar(ji,jj-1) ) / e2v(ji,jj-1) * vmask(ji,jj-1,1)
         jwgt = vmask(ji,jj,1) + vmask(ji-1,jj,1)
         IF ( jwgt == 0 ) jwgt = 1
         grady(ji,jj) = grady(ji,jj) / REAL(jwgt,wp)
      END_2D
      !
   END SUBROUTINE calc_2D_scal_gradient


   SUBROUTINE calc_2D_vec_hdiv( hgt, u, v, hdiv )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_vec_hdiv  ***
      !!
      !! ** Purpose :   Compute horizontal divergence of a 2D velocity field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  u, v   ! input velocities
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  hgt     ! thickness of 2D field
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: hdiv   ! computed divergence
      !
      INTEGER  ::   ji, jj       ! dummy loop indices
      REAL(wp)  :: ztmp1, ztmp2  ! interpolated hgt in u- and v- points
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls-1, nn_hls, nn_hls-1, nn_hls )
         ! i-longitude
         ztmp1 = MIN( hgt(ji+1,jj) , hgt(ji,jj) )
         ztmp2 = MIN( hgt(ji,jj) , hgt(ji-1,jj) )
         hdiv(ji,jj) = u(ji,jj) * e2u(ji,jj) * ztmp1 - u(ji-1,jj) * e2u(ji-1,jj) * ztmp2
         
         ! j-latitude
         ztmp1 = MIN( hgt(ji,jj+1) , hgt(ji,jj) )
         ztmp2 = MIN( hgt(ji,jj) , hgt(ji,jj-1) )
         hdiv(ji,jj) = hdiv(ji,jj) + ( v(ji,jj) * e1v(ji,jj) * ztmp1 - v(ji,jj-1) * e1v(ji,jj-1) * ztmp2 )

         hdiv(ji,jj) = hdiv(ji,jj)  / ( e1e2t(ji,jj)*hgt(ji,jj) )
      END_2D
      !
   END SUBROUTINE calc_2D_vec_hdiv


   SUBROUTINE calc_2D_vec_vort( u, v, vort )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_vec_vort  ***
      !!
      !! ** Purpose :   Compute vertical vorticity of a 2D velocity field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  u, v   ! input velocities
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: vort     ! computed vorticiy
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      REAL(wp), DIMENSION(jpi,jpj) :: zbuf ! working buffer
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         zbuf(ji,jj) = e2v(ji+1,jj) * v(ji+1,jj) - e2v(ji,jj) * v(ji,jj)
         zbuf(ji,jj) = zbuf(ji,jj) - e1u(ji,jj+1) * u(ji,jj+1) + e1u(ji,jj) * u(ji,jj)
         zbuf(ji,jj) = zbuf(ji,jj) * r1_e1e2f(ji,jj) * fmask(ji,jj,1)
      END_2D
      !
      ! set on t-grid
      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
         vort(ji,jj) = 0.25_wp * ( zbuf(ji-1,jj) + zbuf(ji,jj) + zbuf(ji-1,jj-1) + zbuf(ji,jj-1) )
      END_2D
      !
   END SUBROUTINE calc_2D_vec_vort


   SUBROUTINE calc_2D_strain_magnitude( u, v, strain )
      !!----------------------------------------------------------------------
      !!             ***  ROUTINE calc_2D_strain_magnitude  ***
      !!
      !! ** Purpose :   Compute strain magnitude of a 2D velocity field on T-grid
      !!
      !! ** Method  :   * Finite differences
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) ::  u, v   ! input velocities
      REAL(wp), DIMENSION(jpi,jpj), INTENT(out) :: strain ! computed strain
      !
      INTEGER  ::   ji, jj          ! dummy loop indices
      REAL(wp)  :: ztmp             ! local real
      REAL(wp), DIMENSION(jpi,jpj) :: ztrac, zshear ! working arrays
      !!----------------------------------------------------------------------
      !
      DO_2D( nn_hls, nn_hls-1, nn_hls, nn_hls-1 )
         ! expansion rate
         ztmp =   ( u(ji,jj)*r1_e2u(ji,jj) - u(ji-1,jj)*r1_e2u(ji-1,jj) ) * r1_e1t(ji,jj) * e2t(ji,jj) &
              & - ( v(ji,jj)*r1_e1v(ji,jj) - v(ji,jj-1)*r1_e1v(ji,jj-1) ) * r1_e2t(ji,jj) * e1t(ji,jj)  
         ztrac(ji,jj) = ztmp**2 * tmask(ji,jj,1)

         ! shear rate 
         ztmp =   ( u(ji,jj+1)*r1_e1u(ji,jj+1) - u(ji,jj)*r1_e1u(ji,jj) ) * r1_e2f(ji,jj) * e1f(ji,jj) &
              & + ( v(ji+1,jj)*r1_e2v(ji+1,jj) - v(ji,jj)*r1_e2v(ji,jj) ) * r1_e1f(ji,jj) * e2f(ji,jj) 
         zshear(ji,jj) = ztmp**2 * fmask(ji,jj,1)
      END_2D
      !
      ! t-grid
      DO_2D( nn_hls-1, nn_hls-1, nn_hls-1, nn_hls-1 )
         strain(ji,jj) = 0.25_wp * ( zshear(ji-1,jj) + zshear(ji,jj) + zshear(ji-1,jj-1) + zshear(ji,jj-1) )
         strain(ji,jj) = SQRT( strain(ji,jj) + ztrac(ji,jj) ) 
      END_2D
      !
   END SUBROUTINE calc_2D_strain_magnitude


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
