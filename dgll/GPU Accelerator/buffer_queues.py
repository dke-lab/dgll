import os
os.environ['DGLBACKEND'] = 'pytorch'
import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import sklearn.metrics
import time
from dgl.nn import GraphConv
import torch.multiprocessing as mp
# from torch.multiprocessing import Lock, Condition, Process, Queue
# from torch.multiprocessing import Process as Thread
import concurrent.futures
import threading
from queue import Queue
import asyncio
from contextlib import nullcontext

def sample_generator(gpu_queue, condition, train_dataloader, valid_dataloader, model, proc_id):
    d_stream = torch.cuda.Stream()
    best_accuracy = 0
    # generate items
    start = time.time()
    for epoch in range(1):
        with tqdm.tqdm(train_dataloader) as tq:
            for step, (input_nodes, output_nodes, mfgs) in enumerate(tq):
                with torch.cuda.stream(d_stream):
                    with condition:
                        condition.acquire()
                        # print("sample generator got the lock")
                        if gpu_queue.full():
                            # print("queue full in sample generator: goes to sleep")
                            condition.wait()
                            # print("sample generator: After sleep, awakening")
                        gpu_queue.put([mfgs, mfgs[0].srcdata['feat'], mfgs[-1].dstdata['label'], step])
                        # print("sample generator GPU status (after putting item)", torch.cuda.memory_summary() )
                        # divisor = 1024*1024*1024
                        # print("sample generator GPU status (after putting item)", list(map(lambda x: x//divisor, torch.cuda.mem_get_info())) )
                        # print("Item added in sample generator", "Queue Size: ", gpu_queue.qsize())
                        condition.notify()
                        # print("sample_generator broadcat the notification")
                        condition.release()
                        # print("sample_generator released the lock")
        if proc_id == 0:
            model.eval()
            predictions = []
            labels = []
            with tqdm.tqdm(valid_dataloader) as tq, torch.no_grad():
                for input_nodes, output_nodes, mfgs in tq:
                    inputs = mfgs[0].srcdata['feat']
                    labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
                    predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
                predictions = np.concatenate(predictions)
                labels = np.concatenate(labels)
                accuracy = sklearn.metrics.accuracy_score(labels, predictions)
                print('Epoch {} Validation Accuracy {}'.format(epoch, accuracy))
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
    end = time.time()
    print(50*"*")
    print("Accuracy:", best_accuracy*100, "\n")
    print("Time:", end - start)
    with condition:
        condition.acquire()
        gpu_queue.put(None)
        condition.notify()
        condition.release()



def sample_consumer(gpu_queue, condition, opt, model, BUFFER_SIZE = 4):
    con = threading.Condition()
    gradient_buffer = Queue(maxsize= BUFFER_SIZE)
    c_stream = torch.cuda.Stream()
    m_context = model.no_sync
    # m_context = nullcontext
    with torch.cuda.stream(c_stream):
        g_stream = torch.cuda.Stream()
        model.train()
        while True:
            with condition:
                condition.acquire()
                # print("sample generator got the lock")
                if gpu_queue.empty():
                    # print("queue Empty in sample consumer: going to sleep")
                    condition.wait()
                    # print("sample consumer: After sleep, awakening")
                # print("sample consumer GPU status (before getting item)", torch.cuda.memory_summary() )
                # divisor = 1024*1024*1024
                # print("sample consumer GPU status (before getting item)", list(map(lambda x: x//divisor, torch.cuda.mem_get_info())) )
                m_input_lab = gpu_queue.get()
                # print("item got in sample consumer", m_input_lab == None, "Queue Size: ", gpu_queue.qsize())
                condition.notify()
                # print("sample_consumer broadcast the notification")
                condition.release()
                # print("sample_consumer released the lock")

            if m_input_lab == None:
                break
            with m_context():
                opt.zero_grad()
                predictions = model(m_input_lab[0], m_input_lab[1])
                loss = F.cross_entropy(predictions, m_input_lab[2])
                loss.backward()
                with torch.cuda.stream(g_stream):
                    asyncio.run(gradient_generator(model, gradient_buffer, con))
                    asyncio.run(gradient_consumer(model, gradient_buffer, con, opt))
                # opt.step()
                accuracy = sklearn.metrics.accuracy_score(m_input_lab[2].cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
                print(f'>Accuracy', accuracy)
            #
            # if m_input_lab[3] % 3 != 0:
            #     opt.step()
            #     opt.zero_grad()
            #     accuracy = sklearn.metrics.accuracy_score(m_input_lab[2].cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
            #     print(f'>Accuracy', accuracy)
