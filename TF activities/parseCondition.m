addpath('src')
T = readtable('raw/E-GEOD-46903.sdrf.txt');
sourceName = strrep(T.SourceName, ' 1', '');

cellType = T.Comment_Sample_description_;

time = T.FactorValue_TIME_;

stimulation = T.FactorValue_ACTIVATIONSTIMULI_;
stimulation = strrep(stimulation, ' 100U/ml', '');
stimulation = strrep(stimulation, ' 800 u/ml', '');
stimulation = strrep(stimulation, '1�g/ml', '');
stimulation = strrep(stimulation, ' 200U/ml', '');
stimulation = strrep(stimulation, ' ', '');

initialDif = T.FactorValue_INITIALDIFFERENTIATION_;

%unique(cellType)
macroMap = contains(cellType, 'Macrophage');

%unique(T.time)
%[GC, GR] = groupcounts(time(macroMap,:));
timeMap = ismember(time, {'72h'});

%unique(T.time)
%[GC, GR] = groupcounts(initialDif(macroMap,:));
difMap = ismember(initialDif, {'GM-CSF'});

allFilters = and(and(macroMap, timeMap), difMap);

[GC, GR] = groupcounts(stimulation(allFilters,:));
barh(categorical(GR), GC)

sourceName = sourceName(allFilters, :);
stimulation = stimulation(allFilters, :);
for i = 1:length(stimulation)
    stimulation{i} = [stimulation{i} '_' num2str(i)];
end

T = cell2table([sourceName, stimulation], 'VariableNames', {'Condition', 'Stimulation'});
writetable(T, 'results/macrophageKey.txt', 'Delimiter', '\t');