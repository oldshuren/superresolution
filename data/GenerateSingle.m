%%  -*- mode: octave -*-
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate traing data of nanoscopy images  
%
% On Matlab Version 7.5.0342 (R2007b) 
%
%
% 2018-02-22
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function ret=GenerateSingle(points, repeat)
  %% Parameters 
  % Grid size of fluorophore location
  R=10 ; % nm, subpixe size  
  % Camera parameters
  D=100 ;             % nm, pixel size in data frames 
  Kx=7 ; Ky=7 ;       % # of pixels in a data frame
  Lx=D*Kx ; Ly=D*Ky ; % sample is located at [0,Lx]x[0,Ly] nm
  % optical system parameters
  na=1.4 ;    
  lambda=540 ; % nm, wavelength 
  ai=2*pi*na/lambda ; 
  % Calculation of effective sigma of Airy PSF
  rc=7.016/ai ; d=1 ; r=0.01:d:rc ;
  sigma=d*sqrt(8/pi)*sum(besselj(1,ai*r).^2) ; % = 81.2668 nm 
  FWHM=2*sqrt(2*log(2))*sigma ; % = 191.3686 nm 
  % Emitter optics
  Dt=0.01 ;		% seconds, frame period, frame rate = 1/Dt
  Ih=35000 ;  % delta*I*h = total count of photons per emitter
  % SNR 
  % (2) Median SNR
  SPNR=0.2 ;  % mn^2/emitter, signal to Poisson noise ratio
  SGNR=0.3 ;  % mn^2/emitter, signal to Gaussian noise ratio
  b=1e-6*Ih/SPNR ; % =0.1750 photons/s/nm^2, variance of Poisson noise
  G=1e-6*Ih/SGNR ; % =0.1167 photons/s/nm^2, variance of Gaussian noise
  mu=1 ;      % mean of Gaussian noise 
  % Emitter intensity
  I0=Ih ; knowI0=Ih ; % I0 is known

  %% Show training locations of fluorophore 
  Nr=D/R ; % NR^2 = # of subpexils per pixel
  s=zeros(2,1) ; 

  %% Generate training data
  rand('state',0) ;
  randn('state',0) ;
  Cy=10 ; % # of cycles in training ; total # of frames = Cy*Nr^2
  ii=(Kx-1)/2 ; jj=(Ky-1)/2 ; % fluorophore is located at (ii,jj)th pixel
  %figure('Position',[400 400 400 400*(102/108)])
  %%whitebg([0 0 0])

  [input_len tmp] = size(points);
  %printf ('input len %d, repeat %d\n', input_len, repeat);
  len = input_len * repeat;
  %printf ('total len %d, repeat %d\n', len, repeat);
  ret = cell(len,1);
  repeat_cnt = 0;
  for k=1:len,
    if rem(k, input_len) == 0
      repeat_cnt++;
      printf('repeat %d\n', repeat_cnt);
    end
    i = points(rem(k-1, input_len)+1, 1); j= points(rem(k-1, input_len)+1, 2);
    s=R*[j+0.5 ; i+0.5]+D*[jj ; ii] ; % fluorophore location
    V=CCDimg2DGauPSF(sigma,Kx,Ky,D,D,Dt,b,mu,G,s,I0*ones(1,1)) ;
    ret(k) = V;
  end
end


