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

