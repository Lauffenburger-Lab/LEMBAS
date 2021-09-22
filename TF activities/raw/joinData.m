folder = 'E-GEOD-46903.processed.1/';
files = dir(folder);
fileNames = cell(length(files), 1);
matchingFile = false(length(files), 1);
for i = 1:length(files)
   fileNames{i} = files(i).name;
   matchingFile(i) = contains(files(i).name, '.txt');
end
fileNames(not(matchingFile)) = [];
sampleNames = strrep(fileNames, '_sample_table.txt', '');
data = readtable([folder fileNames{1}]);
probeNames = data.ReporterIdentifier;

intensities = zeros(length(probeNames), length(fileNames));
pvals = zeros(length(probeNames), length(fileNames));

warning('off','all')
for i = 1:length(fileNames)
    data = readtable([folder fileNames{i}]);
    intensities(:,i) = table2array(data(:,2));
    pvals(:,i) = table2array(data(:,3));
end
warning('on','all')

tresh = 0.01;
lowestP = min(pvals')';
noRead = (lowestP>tresh);
probeNames(noRead) = [];
intensities(noRead,:) = [];
pvals(noRead,:) = [];

probeMap = containers.Map(probeNames, 1:size(probeNames,1));

opts = detectImportOptions('A-MEXP-1171.adf.txt', 'NumHeaderLines', 18);
opts.VariableTypes(:) = {'char'};
translation = readtable('A-MEXP-1171.adf.txt', opts);
translation = [translation.ReporterName, translation.ReporterDatabaseEntry_entrez_];
noMappedGenes = ismember(translation(:,2), '');
translation(noMappedGenes, :) = [];
overlappingProbe = ismember(translation(:,1), probeNames);
translation(not(overlappingProbe), :) = [];

opts = detectImportOptions('entrez2symbol.txt');
opts.VariableTypes(:) = {'char'};
entrezData = table2cell(readtable('entrez2symbol.txt', opts));
entrezData(ismember(entrezData(:,1), ''), :) = [];
[GC, GR] = groupcounts(entrezData(:,1));
nonUnique = ismember(entrezData(:,1), GR(GC>1));
entrezData(nonUnique,:) = [];
entrezMap = containers.Map(entrezData(:,1), entrezData(:,2));

for i = 1:size(translation, 1)
    if isKey(entrezMap, translation{i,2})
        translation{i,2} = entrezMap(translation{i, 2});
    else
        translation{i,2} = '';
    end
end
noMappedGenes = ismember(translation(:,2), '');
translation(noMappedGenes, :) = [];


uniqueGenes = unique(translation(:, 2));
expression = zeros(size(uniqueGenes, 1), size(intensities, 2));

for i = 1:length(uniqueGenes)
   if mod(i, 1000) == 0
    disp(i/size(uniqueGenes,1))
   end
   
   curGene = ismember(translation(:,2), uniqueGenes{i});
   allProbes = translation(curGene, 1);
   rows = zeros(length(allProbes),1);
   for j = 1:length(allProbes)
       rows(j) = probeMap(allProbes{j});
   end   
   expression(i,:) = mean(intensities(rows, :)); 
end
noVariance = std(expression')<mean(expression')*10^-6;

expression(noVariance,:) = [];
uniqueGenes(noVariance,:) = [];


T = array2table(expression);
T.Properties.RowNames = uniqueGenes;
T.Properties.VariableNames = sampleNames;
writetable(T, 'jointData.txt', 'WriteRowNames', true, 'Delimiter', '\t');
