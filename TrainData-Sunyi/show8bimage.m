% show8bimage(img0,rescale,color,addColormap)
%
% show 8 bit image using selected colormap
%		
%	Input:	img0		-	image to be shown
%					rescale	- ='Yes': The img is scaled to the full range of 8 bits = {0,2^8-1} before save
%										otherwise:	The image is saved as is.
%         color   - ='gray', the 'gray' colormap will be used 
%                 - Otherwise, 'jet' colormap will be used
%    addColormap  - ='No', no colormap will be added to image
%                 - Otherwise, colormap will be added to image
% 
% 4/1/2013
% Yi Sun

function show8bimage(img0,rescale,color,addColormap)

 switch(rescale)
   case 'Yes',
     mx=max(max(img0)) ; mn=min(min(img0)) ;
     img1=uint8(2^8*(img0-mn)/(mx-mn)) ;
   otherwise,
     img1=uint8(img0) ;
 end
 switch(addColormap)
   case 'No',
     img2=img1 ;
   otherwise,
     [R C]=size(img1) ;
     pallet=fix(255*(R-1:-1:0)'/(R-1))*ones(1,ceil(C/20)) ;
     img2=[img1 uint8(pallet)] ;
 end
 switch color
   case 'gray'
     map=colormap(gray(256)) ;
   otherwise
     map=colormap(jet(256)) ;
 end
 image(uint8(img2)) ;
 colormap(map) ;

end

