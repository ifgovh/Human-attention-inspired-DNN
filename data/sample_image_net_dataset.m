cd /project/cortical/RVA-Fractional_motion/data/train/
d = dir('n*');
sample_ind = randperm(length(d),200);
for ii=1:length(sample_ind)
    d_img = dir(fullfile(d(sample_ind(ii)).folder,d(sample_ind(ii)).name,'*JPEG'));
    sample_img_ind = randperm(length(d_img),550);
    eval(['mkdir ','/project/cortical/RVA-Fractional_motion/data/train_sample/ ',d(sample_ind(ii)).name])
    
    for jj=1:500
        eval(['copyfile ',fullfile(d(sample_ind(ii)).folder,d(sample_ind(ii)).name,d_img(sample_img_ind(jj)).name),' ',...
            fullfile('/project/cortical/RVA-Fractional_motion/data/train_sample',d(sample_ind(ii)).name)] )
    end
    eval(['mkdir ','/project/cortical/RVA-Fractional_motion/data/valid_sample/ ',d(sample_ind(ii)).name])
    
    for jj=501:550
        eval(['copyfile ',fullfile(d(sample_ind(ii)).folder,d(sample_ind(ii)).name,d_img(sample_img_ind(jj)).name),' ',...
            fullfile('/project/cortical/RVA-Fractional_motion/data/valid_sample',d(sample_ind(ii)).name)] )
    end
end

