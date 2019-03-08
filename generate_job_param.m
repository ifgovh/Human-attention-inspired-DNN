% generate parameter files for pbs array job
% loss_fun_pool = {'mse','l1','nll','smooth_l1','kl_div','cross_entropy'};
fileID = fopen('job_params_gpu','w');
% for ii = 1:length(loss_fun_pool)
%     for jj = 1:length(loss_fun_pool)
%         fprintf(fileID, [loss_fun_pool{ii},' ',loss_fun_pool{jj},'\n']);
%     end
% end

% for ii = 2.^(1:9)%batch_size
%     for jj = 5:5:25%num_glimpse
%         fprintf(fileID, [num2str(ii),' ',num2str(jj),'\n']);
%     end
% end

for ii = 1:2%batch_size
    if ii == 1
        for jj = 8:4:20%num_glimpse
            fprintf(fileID, ['256 512 ',num2str(jj),'\n']);
        end
    elseif ii == 2
        for jj = 8:4:20%num_glimpse
            fprintf(fileID, ['512 768 ',num2str(jj),'\n']);
        end
    end
end
fclose(fileID);