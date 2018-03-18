baseDir = 'C:\Users\Danny\kaggle\dsbowl18\input\stage1_train';
outputDir = 'C:\Users\Danny\kaggle\dsbowl18\input\train256';
%get all filenames in the training set
allFilesInfo = dir(baseDir);
fileNames = {allFilesInfo.name};

%for a given image get all masks, starts with index = 3
image_boxes = {};
image_names = {};
for i=3:length(fileNames)
    image_folder = strcat(baseDir,'\',fileNames{i},'\images\');
    image_name = strcat(image_folder,fileNames{i},'.png');
    I = imread(image_name);
    I = imresize(I,[256,256]);
    I1 = I(1:128,1:128);
    %figure
    %imshow(I1)
    output_folder = strcat(outputDir,'\',fileNames{i},'.png');
    imwrite(I,output_folder)

end