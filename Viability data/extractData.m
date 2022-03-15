load('preprocessed.mat')
load('mutations.mat')
drugs = {'Erlotinib', 'Lapatinib', 'PD0325901', 'PLX_{4720}', 'CHIR_{265}', 'Selumetinib', 'Vandetanib'};
cell_lines = strrep(cell_lines, 'TUMOR-', '');
conc = [2.5e-3, 8e-3, 2.5e-2, 8e-2, 2.5e-1, 8e-1, 2.53e-0, 8e-0];
concLog = log10(conc * 10^3);

%size(data_fit)
%size(data_test) #No info about which cell-lines are used
%size(data_val)

%tmp=squeeze(data_fit(1,1,1,:,:));
%tmp2=squeeze(data_val(1,1,1,:,:));

nrOfCellines = size(data_fit, 1);
nrOfDrugs = size(data_fit, 2);
nrOfConcentration = size(data_fit, 3);

totalConditions = nrOfCellines*nrOfDrugs*nrOfConcentration;

condition = 1;
testfold = zeros(totalConditions,1);
drugValue = zeros(totalConditions, nrOfDrugs);
viabilityValue = zeros(totalConditions, 1);
cellline = zeros(totalConditions, nrOfCellines);

for i = 1:nrOfCellines
    curTestFold = find(isnan(squeeze(data_fit(i,1,1,:,1))));
    
%     %Add control conditon
%     drugValue(condition, :) = 0;
%     viabilityValue(condition) = 1;
%     celllineNr(condition, i) = 1; 
%     condition = condition + 1;     
    
    for j = 1:nrOfDrugs
        for k = 1:nrOfConcentration
            data = nanmedian(nanmedian(data_fit(i,j,k,:,:)));
            if isnan(data) == false
                drugValue(condition, j) = concLog(k);
                viabilityValue(condition) = data;
                cellline(condition, i) = 1;
                testfold(condition, 1) = curTestFold;
                condition = condition + 1;
            end
        end
    end
end
totalConditions = condition-1;

testfold = testfold(1:totalConditions, :);
drugValue = drugValue(1:totalConditions, :);
viabilityValue = viabilityValue(1:totalConditions, :);
cellline = cellline(1:totalConditions, :);

sampleNames = arrayfun(@num2str, 1:totalConditions, 'UniformOutput', false);        
sampleNames = strcat('C_', sampleNames);   

drugT = array2table(drugValue, 'VariableNames', drugs, 'rowNames', sampleNames);
viabilityT = array2table(viabilityValue, 'VariableNames', {'viability'}, 'rowNames', sampleNames);
foldT = array2table(testfold, 'VariableNames', {'testfold'}, 'rowNames', sampleNames);
celllineT = array2table(cellline, 'VariableNames', cell_lines, 'rowNames', sampleNames);
mutationsT = array2table(mutations, 'VariableNames', cell_lines, 'rowNames', mut_names);


writetable(drugT, 'drug.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');
writetable(viabilityT, 'viability.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');
writetable(foldT, 'CVtest.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');
writetable(celllineT, 'cellLine.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');
writetable(mutationsT, 'mutations.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');



%%
nrOfCellines = size(data_test, 1);
nrOfDrugs = size(data_test, 2);
nrOfConcentration = size(data_test, 3);

totalConditions = nrOfCellines*nrOfDrugs*nrOfConcentration;

condition = 1;
drugValue = zeros(totalConditions, nrOfDrugs);
viabilityValue = zeros(totalConditions, 1);
cellline = zeros(totalConditions, nrOfCellines);

for i = 1:nrOfCellines   
    for j = 1:nrOfDrugs
        for k = 1:nrOfConcentration
            data = nanmedian(nanmedian(data_test(i,j,k,:,:)));
            if isnan(data) == false
                drugValue(condition, j) = concLog(k);
                viabilityValue(condition) = data;
                cellline(condition, i) = 1;
                condition = condition + 1;
            end
        end
    end
end
totalConditions = condition-1;

drugValue = drugValue(1:totalConditions, :);
viabilityValue = viabilityValue(1:totalConditions, :);
%cellline = cellline(1:totalConditions, :); %This information appears to be
%missing

sampleNames = arrayfun(@num2str, 1:totalConditions, 'UniformOutput', false);        
sampleNames = strcat('CIT_', sampleNames);   

drugT = array2table(drugValue, 'VariableNames', drugs, 'rowNames', sampleNames);
viabilityT = array2table(viabilityValue, 'VariableNames', {'viability'}, 'rowNames', sampleNames);
%celllineT = array2table(cellline, 'VariableNames', cell_lines, 'rowNames', sampleNames);

writetable(drugT, 'drugIndependent.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');
writetable(viabilityT, 'viabilityIndependen.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');
%writetable(celllineT, 'cellLine.tsv', 'WriteRowNames', true, 'Delimiter', '\t', 'FileType', 'text');


