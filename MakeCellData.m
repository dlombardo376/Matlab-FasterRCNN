baseDir = 'C:\Users\Danny\kaggle\dsbowl18\input\stage1_train';
croppedDir = 'C:\Users\Danny\kaggle\dsbowl18\input\train256_purpleWhite';
%get all filenames in the training set
allFilesInfo = dir(croppedDir);
fileNames = {allFilesInfo.name};

%for a given image get all masks, starts with index = 3
image_boxes = {};
image_names = {};

allWidth = [];
allHeight = [];
for i=3:length(fileNames)
    image_name = strcat(croppedDir,'\',fileNames{i});
    maskFolder = strcat(baseDir,'\',fileNames{i}(1:(end-4)),'\masks');
    maskInfo=dir(maskFolder);
    maskNames = {maskInfo.name};
    isInitialized = false;
    mask_box = [];
    %mask_box = zeros(length(maskNames)-2,4);
    for j = 3:length(maskNames)
        maskFile = strcat(maskFolder,'\',maskNames{j});
        mask_ = imread(maskFile);
        mask_ = imresize(mask_,[256,256]); %cut into sections of 128x128
        mask1 = mask_(1:256,1:256);
        if(max(mask1(:))>0)
            [mask_row,mask_col] = find(mask1);
            mask_width = (max(mask_row)-min(mask_row));
            mask_height = (max(mask_col)-min(mask_col));
            allWidth = [allWidth;mask_width];
            allHeight = [allHeight;mask_height];
            
            isGood = false;
            if(mask_width > 10 && mask_height > 10 && mask_width < 40 && mask_height < 40) 
                isGood = true; 
            end
            if(isInitialized == false && isGood == true) %first mask we are adding
                mask_box = [min(mask_col) min(mask_row) mask_height mask_width];
                isInitialized = true;
            elseif(isGood==true)
                mask_box = [mask_box; min(mask_col) min(mask_row) mask_height mask_width];
            end
           
        end
    end
    if(length(mask_box>0))
        image_boxes = [image_boxes; mask_box];
        image_names = [image_names; image_name];
        %%read in a sample image
        %I = imread(image_name);
        %I = insertShape(I,'Rectangle',mask_box);
        %figure
        %imshow(I)
    end
end

cellData = [image_names; image_boxes];
cellData = reshape(cellData,[length(image_names),2]);
cellData = cell2table(cellData,'VariableNames',{'imageFilename','cells'});

%make training set
idx = floor(0.6*height(cellData));
trainingData = cellData(1:end,:);
testData = cellData(idx:end,:);