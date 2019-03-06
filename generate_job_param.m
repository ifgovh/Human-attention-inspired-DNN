% generate parameter files for pbs array job
% loss_fun_pool = {'mse','l1','nll','smooth_l1','kl_div','cross_entropy'};
fileID = fopen('job_params','w');
% for ii = 1:length(loss_fun_pool)
%     for jj = 1:length(loss_fun_pool)
%         fprintf(fileID, [loss_fun_pool{ii},' ',loss_fun_pool{jj},'\n']);
%     end
% end

for ii = 2.^(1:9)%batch_size
    for jj = 5:5:25%num_glimpse
        fprintf(fileID, [num2str(ii),' ',num2str(jj),'\n']);
    end
end
fclose(fileID);