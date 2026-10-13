%% 1. تنظیمات محیط
obsInfo = rlNumericSpec([2 1]); 
actInfo = rlNumericSpec([1 1]);
actInfo.LowerLimit = -1; 
actInfo.UpperLimit = 1;

%% 2. ساخت شبکه Critic (اصلاح شده)
% شبکه کریتیک باید State و Action را بگیرد و ترکیب کند.

% مسیر State (مشاهدات)
statePath = [
    imageInputLayer([2 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','fc_state')
    reluLayer('Name','relu_state')
    fullyConnectedLayer(24,'Name','fc_state_out') % خروجی برای ترکیب
    ];

% مسیر Action (اقدام)
actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(24,'Name','fc_action_out') % خروجی برای ترکیب
    ];

% مسیر مشترک (ترکیب دو مسیر با جمع)
commonPath = [
    additionLayer(2,'Name','add') % لایه جمع کننده با 2 ورودی
    reluLayer('Name','relu_common')
    fullyConnectedLayer(1,'Name','CriticOutput')
    ];

% ساخت گراف لایه ها
cLgraph = layerGraph(statePath);
cLgraph = addLayers(cLgraph, actionPath);
cLgraph = addLayers(cLgraph, commonPath);

% اتصال لایه ها (حیاتی)
% وصل کردن خروجی مسیر استیت به ورودی اول جمع کننده
cLgraph = connectLayers(cLgraph,'fc_state_out','add/in1');
% وصل کردن خروجی مسیر اکشن به ورودی دوم جمع کننده
cLgraph = connectLayers(cLgraph,'fc_action_out','add/in2');

% تعریف Critic نهایی
critOpt = rlRepresentationOptions('LearnRate', 0.01, 'GradientThreshold', 1);
critic = rlRepresentation(cLgraph, obsInfo, actInfo, ...
    'Observation',{'state'},'Action',{'action'}, critOpt);

%% 3. ساخت شبکه Actor (بدون تغییر)
aNet = [
    imageInputLayer([2 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(24,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(1,'Name','out')
    tanhLayer('Name','tanh')
    ];

actOpt = rlRepresentationOptions('LearnRate', 0.01, 'GradientThreshold', 1);
actor = rlRepresentation(aNet, obsInfo, actInfo, ...
    'Observation',{'state'},'Action',{'tanh'}, actOpt);

%% 4. تعریف ایجنت و آموزش
agentOptions = rlDDPGAgentOptions(...
    'SampleTime', 0.1, ...
    'DiscountFactor', 0.99, ...
    'MiniBatchSize', 64);

% نویز بالا برای یادگیری سریع (شوک درمانی)
agentOptions.NoiseOptions.Variance = 2.0; 
agentOptions.NoiseOptions.VarianceDecayRate = 1e-3;

agentObj = rlDDPGAgent(actor, critic, agentOptions);

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 300, ...
    'MaxStepsPerEpisode', 100, ...
    'StopTrainingCriteria', 'AverageReward', ...
    'StopTrainingValue', 100000, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% پس از اجرای موفق این کد، دکمه Train را نزنید، بلکه دستور زیر را اجرا کنید:
% train(agentObj, 'rl', trainOpts);
