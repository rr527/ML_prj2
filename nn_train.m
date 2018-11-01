function nn = nn_train(nn,option,train_x,train_y)
    iteration = option.iteration;
    batch_size = option.batch_size;
    m = size(train_x,1);
    num_batches = m / batch_size;
    for k = 1 : iteration
        kk = randperm(m);
        for l = 1 : num_batches
            batch_x = train_x(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            batch_y = train_y(kk((l - 1) * batch_size + 1 : l * batch_size), :);
            
            if strcmp(nn.optimization_method,'RMSPropwNM')
                for kkkk=1:nn.depth-1
                    alpha=0.1;
                    nn.Wtild{kkkk}=nn.W{kkkk};
                    nn.btild{kkkk}=nn.b{kkkk};
                    nn.W{kkkk}=nn.W{kkkk}+alpha*nn.vW{kkkk};
                    nn.b{kkkk}=nn.b{kkkk}+alpha*nn.vb{kkkk};
                end
            end
            nn = nn_forward(nn,batch_x,batch_y);
            nn = nn_backpropagation(nn,batch_y);
            nn = nn_applygradient(nn);
        end
       %disp(['Iteration ' num2str(k) '/' num2str(iteration) ' : ' num2str(t) ' seconds']);
    end
    %figure;
    %plot(nn.cost);grid on;
end