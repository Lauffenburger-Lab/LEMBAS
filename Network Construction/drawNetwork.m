close all
labelIO = true;
modelName = 'KEGGnet';
network = IO([modelName '-Model.tsv']);
annotation = readtable([modelName '-Annotation.tsv'], 'FileType', 'text');
inputNodes = annotation.code(ismember(annotation.ligand, 'True'));
outputNodes = annotation.code(ismember(annotation.TF, 'True'));
uniprot2gene = containers.Map(annotation.code, annotation.name);

[networkList, nodeList] = network2vector(network);
geneNames = nodeList;
for i = 1:length(geneNames)
    if isKey(uniprot2gene, nodeList{i})
        geneNames{i} = uniprot2gene(nodeList{i});
    end
end
geneNames = strrep(geneNames, '_', '-');

G = digraph(networkList(:,1), networkList(:,2));
G.Nodes.Name = nodeList;


colors = get(gca,'colororder');

gray = [0.6 0.6 0.6];

inputNodes = intersect(inputNodes, nodeList);
outputNodes = intersect(outputNodes, nodeList);

in = argwhere(nodeList, inputNodes);
out = argwhere(nodeList, outputNodes);

%referenceConnections = not(isinf(distances(G, inputNodes, outputNodes))) + 0;
%heatmap(geneNames(out), geneNames(in), referenceConnections);
%figure()

fprintf('Interactions:%3.0f\n', size(networkList,1))
fprintf('Nodes:%3.0f\n', size(nodeList,1))
fprintf('Input:%3.0f\n', size(in,1))
fprintf('Output:%3.0f\n', size(out,1))
fprintf('Signaling nodes: %3.0f\n', size(nodeList,1)-size(in,1)-size(out,1))

H = plot(G, 'Layout', 'force', 'UseGravity', true, 'NodeColor', gray, 'EdgeColor', gray, 'LineWidth', 0.5, 'ArrowSize', 5, 'MarkerSize', 3);
xPos = randi(10, length(nodeList), 1); %H.XData
yPos = randi(10, length(nodeList), 1); %H.YData
yPos(in) = 10;
yPos(out) = 1;
layout(H, 'force', 'UseGravity', true, 'XStart', xPos, 'YStart', yPos)
highlight(H, in, 'NodeColor', colors(1,:));
highlight(H, out, 'NodeColor', colors(2,:));
set(gca,'Visible','off')
if labelIO
    labelnode(H, [in; out], [geneNames(in); geneNames(out)]);
end

[effectOfDeletion, deletionList] = getChokepoints(G, inputNodes, outputNodes, 1, 2);
%clustergram(effectOfDeletion, 'RowLabels', outputNodes, 'ColumnLabels', deletionList, 'Symmetric', false, 'Standardize', 'none', 'DisplayRange', 60);

% if labelIO
%     in = argwhere(nodeList, deletionList);
%     highlight(H, in, 'NodeColor', colors(4,:));
%     labelnode(H, in, geneNames(in));
% end
% 
% 
figure()
heatmap(geneNames(argwhere(nodeList, deletionList)), geneNames(argwhere(nodeList, inputNodes)), effectOfDeletion);
filter =sum(effectOfDeletion>10,2)>0;
geneNames(argwhere(nodeList, inputNodes(filter))) 
filter =sum(effectOfDeletion>10,1)>0;
geneNames(argwhere(nodeList, deletionList(filter))) 

figure()
[effectOfDeletion, deletionList] = getChokepoints(G, inputNodes, outputNodes, 1, 1);
heatmap(geneNames(argwhere(nodeList, deletionList)), geneNames(argwhere(nodeList, outputNodes)), effectOfDeletion);
 



function [effectOfDeletion, deletionList] = getChokepoints(G, inputNodes, outputNodes, cutOf, dim)
    nodeList = G.Nodes.Name;
    deletionList = setdiff(nodeList, inputNodes);
    deletionList = setdiff(deletionList, outputNodes);
    in = argwhere(nodeList, inputNodes);
    out = argwhere(nodeList, outputNodes);
    referenceConnections= distances(G, in, out);
    referenceConnections = sum(not(isinf(referenceConnections)), dim);
    effectOfDeletion = zeros(length(referenceConnections), length(deletionList));
    for i = 1:length(deletionList)
        H = rmnode(G, deletionList{i});
        nodeList = H.Nodes.Name;
        in = argwhere(nodeList, inputNodes);
        out = argwhere(nodeList, outputNodes);
        curConnections = distances(H, in, out);
        curConnections = sum(not(isinf(curConnections)), dim);        
        effectOfDeletion(:,i) = referenceConnections-curConnections;
    end
   filter =  sum(effectOfDeletion>cutOf)>cutOf;
   effectOfDeletion = effectOfDeletion(:, filter);
   deletionList = deletionList(filter);
end

function positions = argwhere(nodeList, nodes)
    positions = zeros(length(nodes),1);
    for i = 1:length(nodes)
        positions(i) = find(ismember(nodeList, nodes{i}));
    end
end


function [vector, nodeNames] = network2vector(network)
    sources = network(2:end,1);
    targets = network(2:end,2);
    nodeNames = unique([sources; targets]);

    mapObj = containers.Map(nodeNames, 1:length(nodeNames));
    vector = zeros(length(sources), 2);
    for i = 1:length(sources)
        vector(i,1) = mapObj(sources{i});
        vector(i,2) = mapObj(targets{i});
    end
end

function cellData = IO(fileName)
    opts = detectImportOptions(fileName, 'FileType','text');
    opts = setvartype(opts, opts.VariableNames, 'char');
    T = readtable(fileName, opts);
    headerData = T.Properties.VariableNames;
    cellData = [headerData; table2cell(T)];
end

    