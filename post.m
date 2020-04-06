% %% RGB
% ID = 7;
% start = 'C:\Users\wd_sys\Desktop\DF2020\analysis\s2_validation\s2_validation\ROIs0000_validation_s2_0_p'
% filename = strcat(start,num2str(ID),'.tif')
% I = double(imread(filename));
% I(I>10000) = 10000;
% imshow(I(:,:,[4,3,2])/3000)
% %% VVVH
% start = 'C:\Users\wd_sys\Desktop\DF2020\analysis\s1_validation\ROIs0000_validation_s1_0_p'
% filename = strcat(start,num2str(ID),'.tif')
% I = double(imread(filename));
% I(I<-25) = -25;
% imshow(I(:,:,1),[])
% %% Label
% start = 'C:\Users\wd_sys\Desktop\DF2020\analysis\lc_validation\ROIs0000_validation_lc_0_p'
% filename = strcat(start,num2str(ID),'.tif')
% L = double(imread(filename));
% L = L(:,:,1)
% tabulate(L(:));
% void_classes = [8,9,15];
% valid_classes = [1,2,3,4,5,6,7,10,11,12,14,13,16,17];
% to_classes = [1,1,1,1,1,2,2,4,5,6,6,7,9,10];
% 
% for i=1:3
%     L(find(L==void_classes(i))) = 0;
% end
% 
% for i=1:14
%     L(find(L==valid_classes(i))) = to_classes(i);
% end
% 
% tabulate(L(:));
% ColorI = drawresult(L(:),256,256);
% imshow(ColorI)
% 
% 
% 
% %% Water
% G = I(:,:,3);
% Nir = I(:,:,8);
% NDWI = (G-Nir) ./ (G+Nir);
% %imshow(G,[])
% imshow(NDWI>0.)
% 
% %% Veg
% Red = I(:,:,1);
% Nir = I(:,:,8);
% NDVI = (Nir-Red) ./ (Nir+Red);
% %imshow(G,[])
% imshow(NDVI>0)
% 
% %% Building
% Red = I(:,:,1);
% Mir = I(:,:,12);
% Nir = I(:,:,8);
% NDBI = (Mir-Nir) ./ (Mir+Nir);
% NDVI = (Nir-Red) ./ (Nir+Red);
% %imshow(G,[])
% Build = NDBI-NDVI;
% imshow(NDBI>0.)


%%
Target_class = 8;
mount_idxs=[0,14,41,66,100,126,128,169,242,244,...
               252,271,299,374,375,441,492,637,651,...
              660,672,693,783,787,792,883,919,922,...
              955,968,977,978,982,705,15,84,91,302,...
          412,833,912,953,976];
      
for i=0:5128
    ID = i
    %ID=i
    %disp(ID);
    
    
    start = 'C:\Users\wd_sys\Desktop\DF2020\analysis\lc_0\ROIs0000_test_lc_0_p';
    filename = strcat(start,num2str(ID),'.tif');
    L = double(imread(filename));
    L = L(:,:,1);
    %tabulate(L(:));
    %     void_classes = [8,9,15];
    %     valid_classes = [1,2,3,4,5,6,7,10,11,12,14,13,16,17];
    %     to_classes = [1,1,1,1,1,2,2,4,5,6,6,7,9,10];
    %void_classes = [8,9,15,17];
    valid_classes = [1,2,3,4,5,6,7,8,9,10,11,12,14,13,16,15,17];
    to_classes = [1,1,1,1,1,2,2,3,3,4,5,6,6,7,9,8,10];
    
%     for i=1:4
%         L(find(L==void_classes(i))) = 0;
%     end
    
    for i=1:17
        L(find(L==valid_classes(i))) = to_classes(i);
    end
    
    if sum(L(:)==Target_class)>0 %| Target_class<1000
        tabulate(L(:));
        ColorI = drawresult(L(:),256,256);
        
        start = 'C:\Users\wd_sys\Desktop\DF2020\analysis\test_track1_RGB\ROIs0000_test_RGB_0_p';
        filename = strcat(start,num2str(ID),'.jpg');
        I = double(imread(filename));
        I=I/255*1.5;
        %I(I>10000) = 10000;
        
%         start = 'C:\Users\wd_sys\Desktop\DF2020\analysis\s1_0\ROIs0000_validation_s1_0_p';
%         filename = strcat(start,num2str(ID),'.tif');
%         S = double(imread(filename));
%         S(S<-25) = -25;
        start = 'C:\Users\wd_sys\Desktop\DF2020\test_track1\unet_v1\vis\ROIs0000_test_dfc_0_p';
        filename = strcat(start,num2str(ID),'_vis','.png');
        unet_v1 = double(imread(filename));
        unet_v1=unet_v1/255;
        
        start = 'C:\Users\wd_sys\Desktop\DF2020\test_track1\deeplabv3_v1\vis\ROIs0000_test_dfc_0_p';
        filename = strcat(start,num2str(ID),'_vis','.png');
        deeplabv3_v1 = double(imread(filename));
        deeplabv3_v1=deeplabv3_v1/255;
%         
%         B = I(:,:,2);
%         G = I(:,:,3);
%         R = I(:,:,4);
%         BB = II(:,:,2);
%         GG = II(:,:,3);
%         RR = II(:,:,4);
%         Nir = I(:,:,8);
%         MIR = I(:,:,11);
%         SWIR= I(:,:,12);
        
        
%         NDWI = (G-Nir) ./ (G+Nir);
%         NDSI = (MIR-Nir)./(MIR+Nir);
%         MNDWI=(G-MIR)./(G+MIR);
%         NDVI = (Nir-R) ./ (Nir+R);
%         NDII = (Nir-SWIR) ./ (Nir+SWIR);
%         SIPI = (Nir-B) ./ (Nir+R);
%         MSI = SWIR./Nir;
%         MVI =  (Nir-1.2*R) ./ (Nir+R);
%         NDBI= (SWIR-Nir)./(SWIR+Nir);
%         SR = Nir/R;
%         NDMI = (Nir-MIR) ./ (Nir+MIR);
%         WRI = (G+R)./(Nir+MIR);
%         RRI = B./Nir;
%         NBI = R.*SWIR./Nir;
%         
%         GVI = -0.2728 * II(:,:,1) - 0.2174 * BB - 0.5508 * GG + 0.7721 * RR + 0.0733 * II(:,:,5) - 0.1648 * II(:,:,7);
%         SBI = -0.283 * RR - 0.66 * II(:,:,5) + 0.577 * II(:,:,6) + 0.388 * II(:,:,7);
%         GRABS = (GVI - 0.09178 * SBI + 5.58959);
        
        subplot(2,2,1)
        imshow(I)
        title('rgb')
        subplot(2,2,2)
        imshow(ColorI)
        title('gt')
        subplot(2,2,3)
        imshow(unet_v1)
        title('unet\_v1')
        subplot(2,2,4)
        imshow(deeplabv3_v1)
        title('deeplabv3\_v1')
%         subplot(4,3,3)
%         imshow((S(:,:,2)+25)/25)
%         subplot(4,3,4)
%         imshow((S(:,:,1)+25)/25)
        

%         subplot(4,3,4)
%         imshow(NDII>-0.15);
%         title('NDII')
%         
%         subplot(4,3,5)
%         imshow(NDVI<0.15 & NDVI>0);
%         title('NDVI')
        
%         a=subplot(4,3,6);
%         imshow(NDVI*100,jet);
%         colorbar('Limits',[0,100],'Position',[0.91 0.12 0.05 0.8]);
%         title('NDVI HEAT')
        
%         subplot(4,3,7)
%         imshow(MSI>1.5)
%         title('MSI')
%         
%         subplot(4,3,8)
%         imshow(NBI>750)
%         title('NBI')
        
%         subplot(4,3,9)
%         imshow(SR>0.)
%         title('SR')
        
%         subplot(4,3,9)
%         imshow(GRABS>5.52)
%         title('GRABS')
%         
%         subplot(4,3,10)
%         imshow(NDVI>0.4 & NDVI<0.55)
%         title('ORI')
%         
%         subplot(4,3,11)
%         imshow(NDVI>0.35 & NDVI<0.55 & MSI>1 & MSI<1.5)
%         title('Cropland')
%         
%         subplot(4,3,12)
%         imshow(ColorI)
        
        input('Enter the A:');
    end
end