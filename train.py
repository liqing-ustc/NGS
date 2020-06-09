from utils import *
from dataset import *
from NN_AOG import NNAOG
from diagnosis import ExprTree

import torch
import numpy as np

def train_model(opt):
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.manual_seed)
    train_set = MathExprDataset('train', numSamples=int(10000*opt.data_used), randomSeed=777)
    test_set = MathExprDataset('test')
    print('train:', len(train_set), '  test:', len(test_set))
    model = NNAOG().to(device)
    if opt.pretrain:
        model.sym_net.load_state_dict(torch.load(opt.pretrain))
    train(model, train_set, test_set, opt)

def find_fix(preds, gts, seq_lens, all_probs, nstep):
    etree = ExprTree()
    best_fix_list = []
    for pred, gt, l, all_prob in zip(preds, gts, seq_lens, all_probs):
        pred = pred[:l]
        
        all_prob = all_prob[:l]
        pred_str = [id2sym(x) for x in pred]
        tokens = list(zip(pred_str, all_prob))
        etree.parse(tokens)
        fix = [-1]
        if equal_res(etree.res()[0], gt):
            fix = list(pred)
        else:
            output = etree.fix(gt, n_step=nstep)
            if output:
                fix = [sym2id(x) for x in output[0]]
        best_fix_list.append(fix)
    return best_fix_list

def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    for sample in dataloader:
        img_seq = sample['img_seq']
        label_seq = sample['label_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']
        img_seq = img_seq.to(device)
        label_seq = label_seq.to(device)

        masked_probs = model(img_seq)
        selected_probs, preds = torch.max(masked_probs, -1)
        selected_probs = torch.log(selected_probs+1e-12)
        expr_preds, res_preds = eval_expr(preds.data.cpu().numpy(), seq_len)
        
        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)
        

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    print('Grammar Error: %.2f'%(np.isinf(res_pred_all).mean()*100))
    acc = equal_res(res_pred_all, res_all).mean()

    
    expr_pred_all = ''.join(expr_pred_all)
    expr_all = ''.join(expr_all)
    sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
    
    return acc, sym_acc

def train(model, train_set, test_set, opt):
    mode = opt.mode
    nstep = opt.nstep
    num_workers = opt.num_workers
    batch_size = opt.batch_size
    lr = opt.lr
    reward_decay = opt.decay
    num_epochs = opt.num_epochs
    n_epochs_per_eval = opt.n_epochs_per_eval
    buffer_weight = 0.5

    criterion = nn.NLLLoss(ignore_index=-1)

    params = [{'params': model.parameters()}]
    optimizer = optim.Adam(params, lr=lr)
    
    best_model_wts = deepcopy(model.state_dict())
    best_acc = 0.0
    reward_moving_average = None
    
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers, collate_fn=MathExpr_collate)
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=num_workers, collate_fn=MathExpr_collate)

    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    stats_path = os.path.join(opt.output_dir, "stats_%s_%.2f_%d.json"%(opt.mode, opt.data_used, opt.pretrain != None))
    stats = {
            'train_accs': [],
            'val_accs': []
    }
        
    ###########evaluate init model###########
    acc, sym_acc = evaluate(model, eval_dataloader)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))
    print()
    #########################################
    if mode == "MAPO":
        buffer = [[] for _ in range(len(train_set))]

    iter_counter = -1
    for epoch in range(num_epochs):
        since = time.time()
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        
        model.train()

        if mode == "MAPO":
            counter = 0
            train_queue = []
        for sample in train_dataloader:
            iter_counter += 1

            img_seq = sample['img_seq']
            label_seq = sample['label_seq']
            res = sample['res']
            seq_len = sample['len']
            expr = sample['expr']

            img_seq = img_seq.to(device)
            label_seq = label_seq.to(device)
            max_len = img_seq.shape[1]
            masked_probs = model(img_seq)
            
            if mode == "MAPO":
                 #explore
                m = Categorical(probs = masked_probs)
                preds = m.sample()
                selected_probs = m.log_prob(preds)
                
                rewards = compute_rewards(preds.data.cpu().numpy(), res.numpy(), seq_len)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                
                rewards = rewards - reward_moving_average
                selected_probs = selected_probs.data.cpu().numpy()
                
                j = 0
                for reward in rewards:
                    if reward > 0:
                        flag = 0
                        for buf in buffer[counter]:
                            if buf['preds'] == preds[j].data.tolist():
                                buf['probs'] = np.exp(selected_probs[j].sum())
                                flag = 1
                        if not flag:
                            buffer[counter].append({"preds":preds[j].data.tolist(),"probs":np.exp(selected_probs[j].sum())})
                            
                        total_probs = 0
                        
                        #Re-calculate the weights in buffer
                        for buf in buffer[counter]:
                            total_probs += buf['probs']
                        for buf in buffer[counter]:
                            buf['probs'] = buf['probs']/total_probs
                            
                            train_queue.append({"img_seq":img_seq[j],"expr":expr[j],"res":res[j],"seq_len":seq_len[j],"preds":preds[j],"weights":buf['probs']*buffer_weight,"rewards":reward})
                    counter += 1
                    j += 1                
                
                
                #on-policy
                m = Categorical(probs = masked_probs)
                preds = m.sample()
                selected_probs = m.log_prob(preds) 
                
                rewards = compute_rewards(preds.data.cpu().numpy(), res.numpy(), seq_len)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
                selected_probs = selected_probs.data.cpu().numpy()
                
                j = 0
                for reward in rewards:
                    train_queue.append({"img_seq":img_seq[j],"expr":expr[j],"res":res[j],"seq_len":seq_len[j],"preds":preds[j],"weights":np.exp(selected_probs[j].sum())*(1-buffer_weight),"rewards":reward})
                    j += 1

            elif mode == "BS":
                selected_probs, preds = torch.max(masked_probs, -1)
                selected_probs = torch.log(selected_probs+1e-20)
                masked_probs = torch.log(masked_probs + 1e-20)
                probs = masked_probs

                rewards = compute_rewards(preds.data.cpu().numpy(), res.numpy(), seq_len)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
                
                fix_list = find_fix(preds.data.cpu().numpy(), res.numpy(), seq_len.numpy(), 
                                probs.data.cpu().numpy(), nstep)
                pseudo_label_seq = []
                for fix in fix_list:
                    fix = fix + [-1] * (max_len - len(fix)) # -1 is ignored index in nllloss
                    pseudo_label_seq.append(fix)
                pseudo_label_seq = np.array(pseudo_label_seq)
                pseudo_label_seq = torch.tensor(pseudo_label_seq).to(device)
                loss = criterion(probs.reshape((-1, probs.shape[-1])), pseudo_label_seq.reshape((-1,)))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            elif mode == "RL":
                m = Categorical(probs = masked_probs)
                preds = m.sample()
                selected_probs = m.log_prob(preds)

                rewards = compute_rewards(preds.data.cpu().numpy(), res.numpy(), seq_len)
                if reward_moving_average is None:
                    reward_moving_average = np.mean(rewards)
                reward_moving_average = reward_moving_average * reward_decay \
                        + np.mean(rewards) * (1 - reward_decay)
                rewards = rewards - reward_moving_average
            
                max_len = seq_len.max()
                mask = torch.arange(max_len).expand(len(seq_len), max_len) < seq_len.unsqueeze(1)
                mask = mask.double()
                selected_probs = selected_probs.double()
                selected_probs *= mask.to(device)
                selected_probs = selected_probs.sum(dim=1)
                loss = - torch.tensor(rewards, device=device) * selected_probs
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            selected_probs2, preds2 = torch.max(masked_probs, -1)
            selected_probs2 = torch.log(selected_probs2+1e-12)
            expr_preds, res_pred_all = eval_expr(preds2.data.cpu().numpy(), seq_len)
            acc = equal_res(np.asarray(res_pred_all), np.asarray(res)).mean()
          
            expr_pred_all = ''.join(expr_preds)
            expr_all = ''.join(expr)
            sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
            
            acc = round(acc, 4)
            sym_acc = round(sym_acc, 4)
            stats['train_accs'].append((iter_counter, acc, sym_acc))


        if mode == "MAPO": #learner.start
            batch_number = int(len(train_queue)/batch_size)
            
            for i in range (0, batch_number-1):
                batch_queue = train_queue[i*batch_size:(i*batch_size+batch_size)]

                max_len = 0
                for j in range (0, batch_size):
                    if batch_queue[j]['img_seq'].shape[0] > max_len:
                        max_len = batch_queue[j]['img_seq'].shape[0]
                        
                img_seq = torch.zeros((batch_size,max_len,1,45,45),device=device)
                preds = torch.zeros((batch_size,max_len),device=device)
                seq_len = torch.zeros((batch_size),dtype=torch.long)
                weights = []
                rewards = []
                expr = []
                res = []
                for j in range (0, batch_size):
                    img_seq[j] = batch_queue[j]['img_seq']
                    preds[j] = batch_queue[j]['preds']
                    seq_len[j] = batch_queue[j]['seq_len']
                    weights.append(batch_queue[j]['weights'])
                    rewards.append(batch_queue[j]['rewards'])
                    expr.append(batch_queue[j]['expr'])
                    res.append(batch_queue[j]['res'])
                masked_probs = model(img_seq)
                m = Categorical(probs = masked_probs)
                selected_probs = m.log_prob(preds) 
                mask = torch.arange(max_len).expand(len(seq_len), max_len) < seq_len.unsqueeze(1)
                mask = mask.double()
                selected_probs = selected_probs.double()
                selected_probs *= mask.to(device)
                selected_probs = selected_probs.sum(dim=1)
                loss = - torch.tensor(weights,device=device,dtype=torch.double) * torch.tensor(rewards, device=device,dtype=torch.double) * selected_probs
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                selected_probs2, preds2 = torch.max(masked_probs, -1)
                selected_probs2 = torch.log(selected_probs2+1e-12)
                expr_preds, res_pred_all = eval_expr(preds2.data.cpu().numpy(), seq_len)

                acc = equal_res(np.asarray(res_pred_all), np.asarray(res)).mean()
              
                expr_pred_all = ''.join(expr_preds)
                expr_all = ''.join(expr)
                sym_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])
            
        print("Average reward:", reward_moving_average)
            
        if (epoch+1) % n_epochs_per_eval == 0:
            acc, sym_acc = evaluate(model, eval_dataloader)
            print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))
            if acc > best_acc:
                best_acc = acc
                best_model_wts = deepcopy(model.state_dict())

            acc = round(acc, 4)
            sym_acc = round(sym_acc, 4)
            stats['val_accs'].append((iter_counter, acc, sym_acc))
            json.dump(stats, open(stats_path, 'w'))

                
        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        #print(flush=True)

    acc, sym_acc = evaluate(model, eval_dataloader)
    print('{0} (Acc={1:.2f}, Symbol Acc={2:.2f})'.format('test', 100*acc, 100*sym_acc))
    if acc > best_acc:
        best_acc = acc
        best_model_wts = deepcopy(model.state_dict())
    print('Best val acc: {:2f}'.format(100*best_acc))
    print()
    return    