baseDir = 'C:\Users\danny_000\kaggle\dsbowl18\input\stage1_train';

%get all filenames in the training set
allFilesInfo = dir(baseDir);
fileNames = {allFilesInfo.name};

%for a given image get all masks, starts with index = 3
image_boxes = {};
image_names = {};
for i=3:10%length(fileNames)
    image_folder = strcat(baseDir,'\',fileNames{i},'\images\');
    image_name = strcat(image_folder,fileNames{i},'.png');
    maskFolder = strcat(baseDir,'\',fileNames{i},'\masks');
    maskInfo=dir(maskFolder);
    maskNames = {maskInfo.name};
    mask_box = zeros(length(maskNames)-2,4);
    for j = 3:length(maskNames)
        maskFile = strcat(maskFolder,'\',maskNames{j});
        mask_ = imread(maskFile);
        [mask_row,mask_col] = find(mask_);
        mask_width = (max(mask_row)-min(mask_row));
        mask_height = (max(mask_col)-min(mask_col));
        mask_box(j-2,:) = [min(mask_col) min(mask_row) mask_height mask_width];
    end
    image_boxes = [image_boxes; mask_box];
    image_names = [image_names; image_name];
    %%read in a sample image
    %I = imread(image_name);
    %I = insertShape(I,'Rectangle',mask_box);
    %I=imresize(I,3);
    %figure
    %imshow(I)
end

cellData = [image_names; image_boxes];
cellData = reshape(cellData,[length(image_names),2]);
cellData = cell2table(cellData,'VariableNames',{'imageFileNames','cells'});

%make training set
idx = floor(0.6*height(cellData));
trainingData = cellData(1:idx,:);
testData = cellData(idx:end,:);

%%%%%%%%%%%%%%%%%
%define CNN layers to be used in RCNN
%%%%%%%%%%%%%%%%%
%chose input size 32x32, since all image are greater than 16x16 at least
inputLayer = imageInputLayer([32 32 3]);

%CNN parameters
filterSize = [3 3];
numFilters = 32;

%create the middle layers of the network (conv, relu, max pool)
middleLayers = [
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    convolution2dLayer(filterSize,numFilters,'Padding',1)
    reluLayer()
    maxPooling2dLayer(3,'Stride',2)
    ];

%final layer of the network (dense, softmax loss)
finalLayers = [
    fullyConnectedLayer(64) %64 filters
    reluLayer()
    fullyConnectedLayer(width(cellData)) %num classes + background?
    
    %softmax loss
    softmaxLayer()
    classificationLayer()
    ];

layers = [
    inputLayer
    middleLayers
    finalLayers
    ]

%RCNN has four parts
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);

% Options for step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-5, ...
    'CheckpointPath', tempdir);

% Options for step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);

% Options for step 4.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'InitialLearnRate', 1e-6, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%can load a pretrained model
%change this value to true if training a new model
doTrainingAndEval = true;

if doTrainingAndEval
    %Set random seed to ensure reproducibility
    rng(42);
    
    %train the RCNN object detector
    detector = trainFasterRCNNObjectDetector(trainingData,layers,options,...
        'NegativeOverlapRange',[0 0.3], ...
        'PositiveOverlapRange',[0.6 1], ...
        'BoxPyramidScale',1.2);
end
    
%verify the training
I = imread(cellData.imageFileNames{1});

%run the trained detector on the loaded image
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)