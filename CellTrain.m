%make training set
MakeCellData();

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
    fullyConnectedLayer(2) %num classes + background
    
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

%do training
%Set random seed to ensure reproducibility
rng(42);

%train the RCNN object detector
detector = trainFasterRCNNObjectDetector(trainingData,layers,options,...
    'NegativeOverlapRange',[0 0.3], ...
    'PositiveOverlapRange',[0.6 1], ...
    'BoxPyramidScale',1.2);