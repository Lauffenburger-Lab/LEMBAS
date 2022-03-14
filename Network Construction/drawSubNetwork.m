close all
labelIO = true;
modelName = 'ligandScreen';
network = IO([modelName '-Model.tsv']);
annotation = readtable([modelName '-Annotation.tsv'], 'FileType', 'text');
inputNodes = annotation.code(ismember(annotation.ligand, 'True'));
outputNodes = annotation.code(ismember(annotation.TF, 'True'));
uniprot2gene = containers.Map(annotation.code, annotation.name);
gene2uniprot= containers.Map(annotation.name, annotation.code);


colors = get(gca,'colororder');
gray = [0.6 0.6 0.6];

nrOfNeighbors = 1;

centerNode = gene2uniprot('ELK1')
[networkList, nodeList, sign] = network2vector(network);
inputNodes = intersect(inputNodes, nodeList);
outputNodes = intersect(outputNodes, nodeList);


G = digraph(networkList(:,1), networkList(:,2), sign);
G.Nodes.Name = nodeList;


GUniweight = G;
GUniweight.Edges.Weight(:) = 1;
outDist = distances(GUniweight, centerNode, nodeList)';
inDist = distances(GUniweight, nodeList, centerNode);
removeNodes = not(or(outDist<=nrOfNeighbors, inDist<=nrOfNeighbors));
removeNodes(ismember(nodeList, centerNode)) = false;

G = rmnode(G, nodeList(removeNodes));
inhibition = G.Edges.Weight==-1;
G.Edges.Weight(:) = 1;

subNodeList = table2cell(G.Nodes);
inputNodes = intersect(inputNodes, subNodeList);    
outputNodes = intersect(outputNodes, subNodeList);
in = argwhere(subNodeList, inputNodes);
out = argwhere(subNodeList, outputNodes);    

geneNames = subNodeList;
for i = 1:length(geneNames)
    if isKey(uniprot2gene, subNodeList{i})
        geneNames{i} = uniprot2gene(subNodeList{i});
    end
end    

H = plot(G, 'Layout', 'force', 'UseGravity', true, 'NodeColor', gray, 'EdgeColor', gray, 'LineWidth', 0.5, 'ArrowSize', 5, 'MarkerSize', 3);
highlight(H, in, 'NodeColor', colors(1,:));
highlight(H, out, 'NodeColor', colors(2,:));
highlight(H, centerNode, 'MarkerSize', 10);

highlight(H, 'Edges', inhibition, 'EdgeColor', [0.8, 0, 0]);

labelnode(H, subNodeList, geneNames);
set(gca,'Visible','off')


% for i = 1:length(data)
%    fprintf('%s\t%s\n', data{i},  gene2uniprot(data{i}))
% end

function geneNames = vectorMap(geneNames, uniprot2gene)
    for i = 1:length(geneNames)
        if isKey(uniprot2gene, subNodeList{i})
            geneNames{i} = uniprot2gene(subNodeList{i});
        end
    end    
end


function positions = argwhere(nodeList, nodes)
    positions = zeros(length(nodes),1);
    for i = 1:length(nodes)
        positions(i) = find(ismember(nodeList, nodes{i}));
    end
end


function [vector, nodeNames, sign] = network2vector(network)
    sources = network(2:end,1);
    targets = network(2:end,2);
    nodeNames = unique([sources; targets]);
    activation = ismember(network(2:end,4), '1');
    inhibition = ismember(network(2:end,5), '1');
    sign = activation - inhibition;
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
    