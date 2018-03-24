%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate traing data of nanoscopy images  
%
% On Matlab Version 7.5.0342 (R2007b) 
%
%
% 2018-02-22
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
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
% show fluorophore locaitons in a pixel 
figure('Position',[400 400 400 400*(102/108)])
whitebg([0 0 0])
FPS=10 ; % frames/s
for i=0:Nr-1,
  for j=0:Nr-1,
    t0=cputime ;
    while cputime-t0<1/FPS,
      s=R*[j+0.5 ; i+0.5] ; % fluorophore location in a pixel
      plot(s(1),D-s(2),'w.') ; hold on
      axis([0 D 0 D]) ;
      set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100]) ; 
      set(gca,'YTick',[0 10 20 30 40 50 60 70 80 90 100]) ;
      grid on
      title('100 Training fluorophore locations in a pixel')
      xlabel('(nm)')
      ylabel('(nm)')
    end
    gf=getframe(gcf) ;
  end
end
% show fluorophore locaitons in a frame
ii=(Kx-1)/2 ; jj=(Ky-1)/2 ; % fluorophore is located at (ii,jj)th pixel
figure('Position',[400 400 400 400*(102/108)])
whitebg([0 0 0])
FPS=8 ; % frames/s
for i=0:Nr-1,
  for j=0:Nr-1,
    t0=cputime ;
    while cputime-t0<1/FPS,
      s=R*[j+0.5 ; i+0.5]+D*[jj ; ii] ; % fluorophore location
      plot(s(1),Ly-s(2),'w.') ; hold on
      axis([0 Lx 0 Ly]) ;
      title('100 Training fluorophore locations in a frame')
      xlabel('(nm)')
      ylabel('(nm)')
    end
    gf=getframe(gcf) ;
  end
end

%% Show noiseless training data 
rand('state',0) ;
randn('state',0) ; 
ii=(Kx-1)/2 ; jj=(Ky-1)/2 ; % fluorophore is located at (ii,jj)th pixel
figure('Position',[400 400 400 400*(102/108)])
whitebg([0 0 0])
FPS=8 ; % frames/s
for i=0:Nr-1,
  for j=0:Nr-1,
    t0=cputime ;
    while cputime-t0<1/FPS,
      s=R*[j+0.5 ; i+0.5]+D*[jj ; ii] ; % fluorophore location
      V=CCDimg2DGauPSF(sigma,Kx,Ky,D,D,Dt,b,mu,G,s,100*I0*ones(1,1)) ;
      show8bimage(V,'Yes','gray','No') ;
      title('100 noiseless training data frames') ;
    end
    gf=getframe(gcf) ;
  end 
end

%% Generate training data
rand('state',0) ;
randn('state',0) ;
Cy=10 ; % # of cycles in training ; total # of frames = Cy*Nr^2
TrainData=zeros(Cy*Nr^2,2+Kx*Ky) ;
ii=(Kx-1)/2 ; jj=(Ky-1)/2 ; % fluorophore is located at (ii,jj)th pixel
figure('Position',[400 400 400 400*(102/108)])
whitebg([0 0 0])
FPS=30 ; % frames/s
for k=0:Cy-1 ;
  fprintf(1,'Cycle: %d/%d \n',k+1,Cy) ;
  for i=0:Nr-1,   % row index
    for j=0:Nr-1, % column index
      s=R*[j+0.5 ; i+0.5]+D*[jj ; ii] ; % fluorophore location
      V=CCDimg2DGauPSF(sigma,Kx,Ky,D,D,Dt,b,mu,G,s,I0*ones(1,1)) ;
      TrainData(Cy*k+Nr*i+j+1,1)=i ;  % row index
      TrainData(Cy*k+Nr*i+j+1,2)=j ;  % column index
      for i1=0:Ky-1,    % row index
        for j1=0:Kx-1,  % column index
          TrainData(Cy*k+Nr*i+j+1,2+Ky*i1+j1+1)=V(i1+1,j1+1) ;
        end
      end
      t0=cputime ;
      while cputime-t0<1/FPS,
        show8bimage(V,'Yes','gray','No') ;
        title('10 cycles of 100 training data frames') ;
      end
      gf=getframe(gcf) ;
    end
  end
end
% Save training data 
save TrainData.mat TrainData -MAT
save TrainData.txt TrainData -ASCII -DOUBLE

