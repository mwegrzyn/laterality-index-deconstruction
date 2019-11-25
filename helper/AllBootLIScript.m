myDir = '../data/raw/nii/';

% spm defaults
spm('defaults','fmri');
spm_jobman('initcfg');

dirContent = dir(myDir);
fileList = {dirContent.name};

myFiles = {};
for i=1:length(fileList),
    if (fileList{i}(1)=='t') == 1,
        myFiles = [myFiles strcat(myDir,fileList{i})];
    end
end

% the main script

out = struct('A',char(myFiles),'B1',1,'C1',1,'thr1',-5,'vc',0,'outfile','AllBootLI.txt');
LI(out);