%% WYRÓWNYWANIE HISTOGRAMU (Adaptacyjne wyrównywanie histogramu z ograniczeniem kontrastu)
ImageFolder ='.\all-mias';
OutputFolder = '.\output_histeq';

for i=10:99 % this loop will take 322 pictures and save them in the Matlab folder 
    img = ['\mdb0',num2str(i),'.pgm'];
    fullFileName = fullfile(ImageFolder, img);
    I = imread(fullFileName);
    J = adapthisteq(I);
    imgName = [OutputFolder,'\mdb',num2str(i),'.png'] ;
    imwrite(J,imgName) ; 
end

%% ROZCIĄGANIE HISTOGRAMU
ImageFolder ='.\all-mias';
OutputFolder = '.\output_strech';
for i=100:322 % this loop will take 322 pictures and save them in the Matlab folder 
    img = ['\mdb',num2str(i),'.pgm'];
    fullFileName = fullfile(ImageFolder, img);
    
    I = imread(fullFileName);
    J = imadjust(I,stretchlim(I,0.05),[]);
   
    imgName = [OutputFolder,'\mdb',num2str(i),'.png'] ;
    imwrite(J,imgName) ; 
end
 

%% LAPLACJAN

ImageFolder ='.\all-mias';
OutputFolder = '.\output_laplacjan';
for i=100:322 % this loop will take 322 pictures and save them in the Matlab folder 
    img = ['\mdb',num2str(i),'.pgm'];
    fullFileName = fullfile(ImageFolder, img);
    
    I = imread(fullFileName);
    J =  locallapfilt(I, 0.2, 0.1);
   
    imgName = [OutputFolder,'\mdb',num2str(i),'.png'] ;
    imwrite(J,imgName) ; 
end

%% KONTRAST LOKALNY
ImageFolder ='.\all-mias';
OutputFolder = '.\output_contrast';
for i=10:99 % this loop will take 322 pictures and save them in the Matlab folder 
    img = ['\mdb0',num2str(i),'.pgm'];
    fullFileName = fullfile(ImageFolder, img);
    
    I = imread(fullFileName);
    J = localcontrast(I, 0.4, 0.5);
   
    imgName = [OutputFolder,'\mdb0',num2str(i),'.png'] ;
    imwrite(J,imgName) ; 
end

%% REDUKXJA ZAMGLENIA

ImageFolder ='.\all-mias';
OutputFolder = '.\output_fog';
for i=1:9 % this loop will take 322 pictures and save them in the Matlab folder 
    img = ['\mdb00',num2str(i),'.pgm'];
    fullFileName = fullfile(ImageFolder, img);
    
    I = imread(fullFileName);
    J = imreducehaze(I);
   
    imgName = [OutputFolder,'\mdb00',num2str(i),'.png'] ;
    imwrite(J,imgName) ; 
end

%% konwersja

ImageFolder ='C:\Users\aneta\Desktop\all-mias\all-mias';
OutputFolder = 'C:\Users\aneta\Desktop\all-mias\output_oryginal';
for i=100:322 % this loop will take 322 pictures and save them in the Matlab folder 
    img = ['\mdb',num2str(i),'.pgm'];
    fullFileName = fullfile(ImageFolder, img);
    
    I = imread(fullFileName);
   
    imgName = [OutputFolder,'\mdb',num2str(i),'.png'] ;
    imwrite(I,imgName) ; 
end

%% wykres
subplot(2,3,1)
imshow(I);
title('oryginal');

subplot(2,3,2)
imshow(J1);
title('wyrownanie histogramu');

subplot(2,3,3)
imshow(J5);
title('rozciagniecie histogramu');

subplot(2,3,4)
imshow(J2);
title('kontrast lokalny');

subplot(2,3,5)
imshow(J3);
title('filtracja laplacjanem');

subplot(2,3,6)
imshow(J4);
title('redukcja "zamglenia"');


%% Low pass

[M, N] = size(I);
FT_img = fft2(double(I));

D0 = 50; 
u = 0:(M-1);
idx = find(u>M/2);
u(idx) = u(idx)-M;
v = 0:(N-1);
idy = find(v>N/2);
v(idy) = v(idy)-N;

[V, U] = meshgrid(v, u);

D = sqrt(U.^2+V.^2);
H = double(D <= D0);
G = H.*FT_img;

output_image = real(ifft2(double(G)));

%%
% Displaying Input Image and Output Image
figure()
subplot(1, 3, 1), imshow(I), title('oryginał');
subplot(1, 3, 2), imshow(output_image30, [ ]), title('low pass 30');
subplot(1, 3, 3), imshow(output_image50, [ ]), title('low pass 50');
