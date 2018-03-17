baseDir = 'C:\Users\Danny\kaggle\dsbowl18\input\stage1_train';

%get all filenames in the training set
allFilesInfo = dir(baseDir);
fileNames = {allFilesInfo.name};

%for a given image get all masks, starts with index = 3
image_boxes = {};
image_names = {};
for i=3:15%length(fileNames)
    image_folder = strcat(baseDir,'\',fileNames{i},'\images\');
    image_name = strcat(image_folder,fileNames{i},'.png');
    maskFolder = strcat(baseDir,'\',fileNames{i},'\masks');
    maskInfo=dir(maskFolder);
    maskNames = {maskInfo.name};
    isInitialized = false;
    %mask_box = zeros(length(maskNames)-2,4);
    for j = 3:length(maskNames)
        maskFile = strcat(maskFolder,'\',maskNames{j});
        mask_ = imread(maskFile);
        mask_ = imresize(mask_,[128,194]);
        if(max(mask_(:))>0)
            [mask_row,mask_col] = find(mask_);
            mask_width = (max(mask_row)-min(mask_row));
            mask_height = (max(mask_col)-min(mask_col));
            isGood = false;
            if(mask_width > 0 && mask_height > 0) isGood = true; end
            if(isInitialized == false && isGood == true) %first mask we are adding
                mask_box = [min(mask_col) min(mask_row) mask_height mask_width];
                isInitialized = true;
            elseif(isGood==true)
                mask_box = [mask_box; min(mask_col) min(mask_row) mask_height mask_width];
            end
        end
    end
    image_boxes = [image_boxes; mask_box];
    image_names = [image_names; image_name];
    %%read in a sample image
%     I = imread(image_name);
%     I = imresize(I,[128,194]);
%     I = insertShape(I,'Rectangle',mask_box);
%     figure
%     imshow(I)
end

cellData = [image_names; image_boxes];
cellData = reshape(cellData,[length(image_names),2]);
cellData = cell2table(cellData,'VariableNames',{'imageFilename','cells'});

%make training set
idx = floor(0.6*height(cellData));
trainingData = cellData(1:end,:);
testData = cellData(idx:end,:);