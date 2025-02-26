import numpy as np

import torch
from torch.autograd import Variable
import torch.autograd as autograd
from torch.nn.utils import clip_grad_norm_
from torchmetrics.functional.regression import  r2_score, \
                                                mean_squared_error, \
                                                mean_absolute_percentage_error, \
                                                mean_absolute_error

from tqdm import tqdm
from IPython.display import clear_output
import time
import wandb
import os
import matplotlib.pyplot as plt
from .BertImpl import Bert_Test
_wandb = False
from .__vocabs import dna_vocab_revers
import gdown
import urllib.request
import subprocess
# 

def download_demo_resources():
    try:
        os.mkdir("./data",)
    except OSError as error:
        pass
    print("Downloading hg19")
    urllib.request.urlretrieve("http://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz", "./data/hg19.fa.gz")
    process_hg19 = subprocess.Popen("gzip -d ./data/hg19.fa.gz", shell = True)
    print("Downloading hg38")
    urllib.request.urlretrieve("https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz", "./data/hg38.fa.gz")
    process_hg38 = subprocess.Popen("gzip -d ./data/hg38.fa.gz", shell = True)
    print("Downloading demo resources")
    gdown.download_folder(id="https://drive.google.com/drive/folders/17_IMgmn_QZ_E52vxntmTfUU3eXHy-E2G", output="./data")
    process_hg19.wait()
    process_hg38.wait()


def WandbLogging(LogData,StartTime, Delay):
    NowTime = time.perf_counter()
    if NowTime - StartTime > Delay:
        wandb.log(LogData)
        return NowTime
    else:
        return StartTime

def generate_test_fasta(generator,n_count,val_ex = None,val_count = 0, path = None, cuda = None, filter = None):
    '''
        Function to generate test data set during model learning
    '''
    if not path:
        path = "./test.fasta"
    max_rand = len(val_ex)
    while val_count > 0:
                num = int(np.random.randint(max_rand, size=1))
                OheSeq = val_ex[num][0]              
                filt = OheSeq[0][0]
                OheSeq = OheSeq[0][0].detach().cpu().numpy()
                Seq = [dna_vocab_revers[one] for one in np.argmax(OheSeq,1)]
                if filter:
                    #TODO: naxer?
                    if filter(filt):
                        with open(path, 'a+') as f:
                            f.write('>%s_val\n%s\n' % (n_count+val_count, ''.join(Seq)))
                        val_count-=1
                        if val_count == 0:
                             break
                else:
                    with open(path, 'a+') as f:
                            f.write('>%s_val_no_filter\n%s\n' % (n_count+val_count, ''.join(Seq)))
                    val_count-=1
                    if val_count == 0:
                             break

    while n_count > 0:
        noise = Variable(torch.randn((1200,1))).cuda()
        filt = generator(noise)
        OheSeq = filt.detach().cpu().numpy()
        Seq = [dna_vocab_revers[one] for one in np.argmax(OheSeq,0)]
        if filter:
            if filter(filt):
                with open(path, 'a+') as f:
                    f.write('>%s\n%s\n' % (n_count, ''.join(Seq)))
                    n_count-=1
        else:
            with open(path, 'a+') as f:
                    f.write('>%s\n%s\n' % (n_count, ''.join(Seq)))
                    n_count-=1


def regression_learning(model,optimizer, scheduler, criterion,train_loader,lr = 1e-4, n_epoch = 400, verbose = False):
    epoch_num = n_epoch
    loss_arr = []
    for epoch in tqdm(range(epoch_num)):
        model.train()
        epoch_loss = 0
        for data in train_loader:
            x, y = data
            optimizer.zero_grad()
            output = model(x[0].reshape([1,-1]))
            loss = criterion(output.float(), y.float())
            loss.backward()
            optimizer.step()
            epoch_loss+=loss
        scheduler.step()
        loss_arr.append(epoch_loss.cpu().detach().numpy()/len(train_loader))
        if verbose:
            clear_output()
            plt.plot(loss_arr)
            plt.show()

    return model
        


def regression_val(model, test_loader,train_loader):
    '''
        Function to print model learning log
    '''
    pred_test = []
    real_test = []
    pred_train = []
    real_train = []
    with torch.no_grad(): 
        for one in test_loader:
            x,y = one
            output = model(torch.reshape(x,(1,-1)))
            pred_test.append(output[0][0].to('cpu'))
            real_test.append(y[0].to('cpu'))
            del x,y,output
            torch.cuda.empty_cache()
        for one in train_loader:
            x,y = one
            output = model(torch.reshape(x,(1,-1)))
            pred_train.append(output[0][0].to('cpu'))
            real_train.append(y[0].to('cpu'))
            del x,y,output
            torch.cuda.empty_cache()
        pred_test = torch.Tensor(pred_test,device= 'cpu')
        real_test = torch.Tensor(real_test, device= 'cpu')
        pred_train = torch.Tensor(pred_train,device= 'cpu')
        real_train = torch.Tensor(real_train, device= 'cpu')
        print(f"R2_test = {r2_score(pred_test,real_test)} \n\
                RMSE_test = {torch.sqrt(mean_squared_error(pred_test,real_test))} \n\
                MAPE_test = {mean_absolute_percentage_error(pred_test,real_test)}\n\
                MAE_test = {mean_absolute_error(pred_test,real_test)}\n\
                R2_train = {r2_score(pred_train,real_train)}\n\
                RMSE_train = {torch.sqrt(mean_squared_error(pred_train,real_train))}\n\
                MAPE_train = {mean_absolute_percentage_error(pred_train,real_train)}\n\
                MAE_train = {mean_absolute_error(pred_train,real_train)}\n")
        torch.cuda.empty_cache()


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).cuda()
        # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(1,1200).fill_(1.0), requires_grad=False).cuda()
        # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def reset_grad(model):
    for param in model.parameters():
        param.grad = None


def get_genomic_data_stats(genomicData, verbose = False):
    hist_real = np.zeros(genomicData[0].T.shape, dtype = int)
    names = ["A","T","G","C","N"]
    for i in genomicData:
        for pos, nucl in enumerate(torch.argmax(i.cpu(), dim = 1)):
            hist_real[nucl][pos] += 1
    nucl_hits_natural = { name: sum(hist_real[i]) for i,name in enumerate(names)}

    if verbose:
        for name, stat, c in zip(names, hist_real, ["r","g","b","violet","y"]):
            plt.ylabel("Hits")
            plt.xlabel("Position")
            plt.title(name)
            plt.bar(range(len(stat)),stat, color = c)
            plt.show()
        x, y = zip(*nucl_hits_natural.items()) # unpack a list of pairs into two tuples
        plt.title("Nucleotide summary")
        plt.bar(x, y)
        plt.show()

    return (hist_real, nucl_hits_natural)

# Bert

def draw_and_safe_generator_GC_content(generator, hist_real, nucl_hits_natural, seq_len = 100, n = 1000, filename = "bars", show = False, save = True): # d[0].t shape
    plt.ioff()
    clear_output()
    names = ["A","T","G","C","N"]
    hist = np.zeros(hist_real.shape)
    for i in range(n):
            out = generator(torch.randn(seq_len).cuda())[0]
            for pos, nucl in enumerate(torch.argmax(out, dim = 0)):
                hist[nucl][pos] += 1

    nucl_hits_generic = {name: sum(hist[i]) for i,name in enumerate(names)}


    for truth,gen, name in zip(hist_real, hist, names):
        # Define the positions
        positions = np.arange(1, len(truth) + 1)

        # Set the figure size
        fig = plt.figure(figsize=(12, 8))  

        # Plot each dataset using bar with transparency
        width = 1  # Width of each bar

        plt.bar(positions, truth/max(truth), width=width, color='m', alpha=0.4, label='Truth')
        plt.bar(positions, gen / max(gen), width=width, color='b', alpha=0.4, label='Generic')

        plt.text(0.05, 0.95, f'{name} hits sum in {n} variants : {nucl_hits_generic[name]}', transform=plt.gca().transAxes, color='black', fontsize=12)
        plt.text(0.05, 0.90, f'{name} hits sum in natural : {nucl_hits_natural[name]}', transform=plt.gca().transAxes, color='b', fontsize=12)
        # Set the labels and title
        plt.xlabel('Position')
        plt.ylabel('Hits')
        plt.title(f'Histogram Plotof {name} nucl')

        # Add legend
        plt.legend()

        if save:
            plt.savefig(filename+name+".png")

        # Display the plot
        if show:
            plt.show()
        else:
            plt.close(fig)

    return nucl_hits_generic




def reset_grad(model):
    for param in model.parameters():
        param.grad = None



def GAN_learning_ZDNA(generator,discriminator,optimizer_D, optimizer_G ,dataloader, hist_real, nucl_hits_natural, n_epochs = 6, lr = 1e-4, filter = None):
    
    clear_output()
    # suggested default = 200
    history = []
    epoch_tqdm_bar = tqdm(range(n_epochs))
    GC_check = []
    for epoch in epoch_tqdm_bar:
        mean_loss_gen = 0
        mean_loss_cr = 0
        # dataloader = DataLoader(GenData,6,drop_last=True, shuffle = True)
        StartTime = time.perf_counter()
        
        for count,data in enumerate(dataloader):
            #   Critical forward-loss-backward-update
            if count % 4 != 0:

                # Sample data
                noise = Variable(torch.randn(data.shape)).cuda()# Random sampling Tensor(batch_size, latent_dim) of Gaussian distribution

                real_data = Variable(data[0].T)
                # Sintetic                
                fake_data = generator(noise)
                critics_real = discriminator(real_data)
                critics_fake = discriminator(fake_data)
                critics_loss = -torch.mean(critics_real) + torch.mean(critics_fake)   
                critics_loss.backward()
                optimizer_D.step()

                # # Weight clipping
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                # Housekeeping - reset gradient
                reset_grad(generator)
                reset_grad(discriminator)

                with torch.no_grad():
                    mean_loss_cr+=critics_loss.item()
            else:
            #   Generator forward-loss-backward-update
            
                noise = Variable(torch.randn(data.shape)).cuda() # Random sampling Tensor(batch_size, latent_dim) of Gaussian distribution
                real_data = Variable(data[0].T)

                fake_data = generator(noise)
                critics_fake = discriminator(fake_data)

                generator_loss = -torch.mean(critics_fake)
                
                generator_loss.backward()
                optimizer_G.step()

                # Housekeeping - reset gradient
                reset_grad(generator)
                reset_grad(discriminator)
                with torch.no_grad():
                    mean_loss_gen+=generator_loss.item()

            if _wandb:
                table = wandb.Histogram(np_histogram = np.histogram(np.argmax(generator(noise).detach().cpu().numpy(),0)))
                WandbLogDict = {"Epoch": epoch+1, "Critics_loss": critics_loss.item(), "Generator_loss": generator_loss.item(),"Results": table}
                StartTime = WandbLogging(WandbLogDict, StartTime, 120) 
            if count % 100 == 0:
                history.append(Bert_Test(generator, data, 100)["rate"])
            if count % 1000==0:
                GC_check.append(draw_and_safe_generator_GC_content(generator, hist_real,nucl_hits_natural, data.shape, filename= f"../def/generator_learning/{count}_e{epoch}_" ))
        check = Bert_Test(generator, data, 500)
        rate = check["rate"]
        clear_output()
        plt.plot(history)
        plt.show()
        G = []
        C = []
        for d in GC_check:
            G.append(d["G"])
            C.append(d["C"])
        plt.plot(G)
        plt.plot(C)
        plt.savefig(f"../def/generator_learning/GC_content_{epoch}_epoch.png")
        plt.show()

        epoch_tqdm_bar.set_description(
f"Mean train loss of epoch {epoch} :\
gen = {mean_loss_gen / count * 4 / 3} \
cr = {mean_loss_cr / count * 4} \
Bert rate = {rate}")    
        if epoch % 10 == 0:
            generate_test_fasta(generator,5,dataloader,0,path = f"./test_{int(epoch)}.fasta")
            # optimizer_G.lr = lr*0.01
            # optimizer_D.lr = lr*0.01
        if epoch == int(n_epochs/2):
            generate_test_fasta(generator,5,dataloader,0,path = f"./test_{int(n_epochs/3)}.fasta")
            # optimizer_G.lr = lr*0.01
            # optimizer_D.lr = lr*0.01
        if epoch == int(n_epochs/3 * 2):
            generate_test_fasta(generator,5,dataloader,0,path = f"./test_{int(n_epochs/2)}.fasta")
            # optimizer_G.lr = lr*0.000001
            # optimizer_D.lr = lr*0.000001

    #generate_test_fasta(generator,5,dataloader,5,path = f"./ER_test_NonFiltred.fasta")
    return (generator,discriminator,GC_check)
