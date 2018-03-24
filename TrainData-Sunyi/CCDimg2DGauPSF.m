% V=CCDimg2DGauPSF(sigma,Kx,Ky,Dx,Dy,Dt,b,mu,G,xy,Ih)  
%
% Produce CCD image with 2D Gaussian PSF from emitter locations in xy, and intensities in Ih 
% 
% See [1] for technical details
% [1] Sun, Y. Localization precision of stochastic optical localization nanoscopy using single
% frames. J. Biomed. Optics 18, 111418.1-15 (2013).
% 
% Input:
%   sigma   - SDV of Gaussian PSF 
%		Kx, Ky	- Image size is Ky*Kx in pixels, sample is located at [0,Kx*Dx]x[0,Ky*Dy]
%		Dx, Dy	- Pixel size is Dx*Dy square nanometers (nm^2)
%   Dt   		- imaging time in second
%   b       - number of Poisson noise photons /s/nm^2 
%   mu      - mean of Gaussian noise w(t,x,y) in voltage/s/nm^2, 
%             mean of Gaussian noise in each pixel is mu_w=Dx*Dy*Dt*mu 
%   G       - variance of w(t,x,y)
%           - variance in each pixel is sigma_w^2=Dx*Dy*Dt*G 
%   xy      - ith colume of xy is (x,y) coordinates of ith emitter location 
%   Ih		  - ith element is the mean number of detected photons per second from ith emitter 
%
% Output:
%   V       - CCD image with shot noise, Poisson noise, and Gaussian noise
%
% Note:       All distances are in nm. 
%
% 7/11/2013
% Yi Sun

function V=CCDimg2DGauPSF(sigma,Kx,Ky,Dx,Dy,Dt,b,mu,G,xy,Ih) 

[tmp M]=size(xy) ;
if M<1, 
  fprintf(1,'# of dyes is zero. \n') ;
  return ;
end
v=zeros(Ky,Kx) ; 
ky=0:Ky-1 ; kx=0:Kx-1 ;
for i=1:M,
  Dxk1=(Dx*(kx+1)-xy(1,i))/sigma ; Dxk0=(Dx*kx-xy(1,i))/sigma ; 
	Dyk1=(Dy*(ky+1)-xy(2,i))/sigma ; Dyk0=(Dy*ky-xy(2,i))/sigma ; 
	qx=(Qfunc(-Dxk1)-Qfunc(-Dxk0)) ;
	qy=(Qfunc(-Dyk1)-Qfunc(-Dyk0)) ;
	v=v+Ih(i)*(Dx*Dy)^(-1)*qy'*qx ;
end
v=Dt*Dx*Dy*(v+b) ;
V=poissrnd(v) ;
V=V+Dt*Dx*Dy*mu+sqrt(Dt*Dx*Dy*G)*randn(size(V)) ;

end
