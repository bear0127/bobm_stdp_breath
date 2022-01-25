import os
import numpy as np
import matplotlib.pyplot as plt
from brian2 import *
import brian2 as b2
import math
from scipy import signal


# 找样本函数无递归版
def findfile(path, file_last_name):
    file_name = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        # 如果是文件夹，则递归
        if os.path.isdir(file_path):
            print('进入子文件夹')
            for sub_file in os.listdir(file_path):
                sub_file_path = os.path.join(file_path, sub_file)
                if os.path.splitext(sub_file_path)[1] == file_last_name:
                    file_name.append(sub_file_path)
                else:
                    pass
        elif os.path.splitext(file_path)[1] == file_last_name:
            file_name.append(file_path)
    return file_name


# 数据归一化：只去基线
def normalize(values):
    for i in range(len(values)):
        row_min_value = np.min(values[i])
        for j in range(len(values[i])):
            values[i][j] -= row_min_value
    normalized_data = values
    return normalized_data


# BSA脉冲编码器：模拟信号 -> 脉冲发放时刻
def BSA_encoder(input_data, filter, threshold):
    spike_time_list = []
    # 复制一份数据进行处理，避免修改原始数据
    input_data_copy = np.copy(input_data)
    for i in range(len(input_data_copy)):
        error1 = 0
        error2 = 0
        for j in range(len(filter)):
            if (i + j - 1) <= len(input_data_copy) - 1:
                error1 += abs(input_data_copy[i + j - 1] - filter[j])
                error2 += abs(input_data_copy[i + j - 1])

        if (error2 - error1) >= threshold:
            spike_time_list.append(i)

            for j in range(len(filter)):
                if (i + j - 1) <= len(input_data_copy) - 1:
                    input_data_copy[i + j - 1] -= filter[j]
        else:
            continue
    return spike_time_list


# 突触连接可视化
def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), 'ok', ms=10)
    plot(ones(Nt), arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], '-k')
    xticks([0, 1], ['Source', 'Target'])
    ylabel('Neuron index')
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, 'ok')
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel('Source neuron index')
    ylabel('Target neuron index')


#------------------------------------------------------------------------------
# read and prepare all data for training
#------------------------------------------------------------------------------

# 读取数据集文件夹下所有子文件夹中的样本数据
txt_folder_path = './data/train_val/'
# 使用BSA脉冲编码求出传感器响应曲线的脉冲发放时间矩阵
# def get_spikes_time_matrix(txt_folder_path):
txt_file_path = findfile(txt_folder_path, '.txt')
print(txt_file_path)
norm_data_list = []
for txt_file in txt_file_path:
    data_per_file = [[] for i in range(2)]
    with open(txt_file) as f:
        content = f.readlines()
        row = 0
        for items in content:
            row_data = items.split()
            for response_value in row_data:
                data_per_file[row].append(float(response_value))
            row += 1
    # print(data_per_file)
    norm_data = normalize(np.array(data_per_file))
    norm_data_list.append(norm_data)
norm_data_list = np.array(norm_data_list)
# print(norm_data_list)
print(norm_data_list.shape)

# 打乱数据先后顺序
shuffle_indices = np.arange(norm_data_list.shape[0])
np.random.shuffle(shuffle_indices)
norm_data_shuffle_list = norm_data_list[shuffle_indices]
# print(norm_data_shuffle_list)
print(norm_data_shuffle_list.shape)

# 对数据进行脉冲编码
sensor_numbers = 2
sampling_time = 295
# 仿真时间
run_time = 5000 * ms
# 时间缩放因子k
scaling_factor_time = run_time / sampling_time
sensor_indices_copy_list = []
input_spike_times_list = []
for norm_data in norm_data_shuffle_list:
    # 对归一化后的数据进行去噪处理
    for col_data in norm_data:
        # 去掉递减的响应曲线
        # if col_data[0] > (col_data[len(col_data)//2]+0.25):
        #     for j in range(len(col_data)):
        #         col_data[j] = 0
        # 去除响应曲线中的毛刺
        # else:
        for j in range(1, len(col_data)-1):
            if col_data[j] > (col_data[j-1] * 1.25) and col_data[j] > (col_data[j+1] * 1.25):
                col_data[j] = col_data[j-1]
            elif col_data[j] < (col_data[j-1] / 1.2) and col_data[j] < (col_data[j+1] / 1.2):
                col_data[j] = col_data[j - 1]
            else:
                continue

    # 查看去噪后的传感器响应曲线
    sampling_points = range(1, len(norm_data[0]) + 1)
    # plt.figure()
    # k = 1
    # for col_data in norm_data:
    #     plt.plot(sampling_points, col_data, label='sensor_%s' % k)
    #     k += 1
    # plt.legend()

    # 脉冲发放时间矩阵spikes_time_matrix
    spikes_time_matrix = []
    # 对norm_data采用BSA脉冲编码，输出2通道的脉冲发放时刻
    # firwin(滤波器宽度, [a, b])
    # a 和 b 的值越小，编码出来的脉冲发放频率越高
    # 为保证脉冲发放的连续性，b 不能比 a 大太多
    FIR_filter = signal.firwin(45, [1 / 295, 8 / 295], window='hamming', pass_zero=False)
    for row in range(len(norm_data)):
        BSA_spike_time_list = BSA_encoder(norm_data[row], FIR_filter, threshold=0.9550)
        spikes_time_matrix.append(BSA_spike_time_list)
    print(spikes_time_matrix)

    # 将脉冲发放时刻转换为嗅球模型的输入
    sensor_indices = [[] for i in range(2)]
    spike_times_list = [[] for i in range(2)]

    row = 0
    for row in range(len(spikes_time_matrix)):
        for col in range(len(spikes_time_matrix[row])):
            if spikes_time_matrix[row][col] != None:
                spike_times_list[row].append(spikes_time_matrix[row][col])
    # print(spikes_time_list)
    # 同一个神经元的输入脉冲时刻不能存在相同值
    for i in range(len(spike_times_list)):
        # 对同一个神经元的输入脉冲时刻去重
        spike_times_list[i] = list(set(spike_times_list[i]))
        for j in range(len(spike_times_list[i])):
            sensor_indices[i].append(i)
    # print(sensor_indices, '\n', spike_times_list)
    # 将二维矩阵转为一维以适应嗅球模型的输入
    sensor_indices_copy = []
    spike_times_list_copy = []
    for sensor_indice in sensor_indices:
        for element in sensor_indice:
            sensor_indices_copy.append(element)
    for spike_times in spike_times_list:
        for spike_time in spike_times:
            spike_times_list_copy.append(spike_time)
    print(sensor_indices_copy, '\n', spike_times_list_copy)
    # 将每个样本的脉冲输入时刻input_spike_times和对应的神经元序号sensor_indices_copy添加到list中
    sensor_indices_copy_list.append(sensor_indices_copy)
    input_spike_times = spike_times_list_copy * scaling_factor_time
    input_spike_times_list.append(input_spike_times)
print('sample numbers: ', len(sensor_indices_copy_list), len(input_spike_times_list))

    # plt.figure()
    # plt.scatter(input_spike_times, sensor_indices_copy)
    # plt.show()

#------------------------------------------------------------------------------
# set the neurons and connections
#------------------------------------------------------------------------------

input_groups = {}
neuron_groups = {}
connections = {}
state_monitors = {}
spike_monitors = {}

eqs1 = '''
dv/dt = (int(v >= vr) * (k1 * (v - vr) * (v - vt) - u + I) + int(v < vr) * (vr - v + 300))/C/tau : 1
dI/dt = -k2 * I / tau : 1
du/dt = a * (b * (v - vr) - u)/tau : 1
vr : 1
vt : 1
k1 : 1
k2 : 1
C : 1
a : 1
b : 1
tau : second
'''

# spike generator
input_groups['input_neurons'] = SpikeGeneratorGroup(sensor_numbers, indices=[0], times=[0*ms])

# ORN
neuron_groups['ORN'] = NeuronGroup(2, eqs1, threshold='v>35', reset='''v = -50; u += 200''', method='euler')
neuron_groups['ORN'].v = -55
neuron_groups['ORN'].vr = -55
neuron_groups['ORN'].vt = -50
neuron_groups['ORN'].k1 = 1
neuron_groups['ORN'].k2 = 0.75
neuron_groups['ORN'].C = 25
neuron_groups['ORN'].a = 0.4
neuron_groups['ORN'].b = 2.6
neuron_groups['ORN'].tau = 25*ms

# MC
neuron_groups['MC'] = NeuronGroup(2, eqs1, threshold='v>35', reset='''v = -50; u += 200''', refractory=1*ms, method='euler')
neuron_groups['MC'].I = 0
neuron_groups['MC'].v = -55
neuron_groups['MC'].vr = -55
neuron_groups['MC'].vt = -50
neuron_groups['MC'].k1 = 1
neuron_groups['MC'].k2 = 3.75
neuron_groups['MC'].C = 40
neuron_groups['MC'].a = 0.4
neuron_groups['MC'].b = 2.6
neuron_groups['MC'].tau = 25*ms

# GC
neuron_groups['GC'] = NeuronGroup(2, eqs1, threshold='v>35', reset='''v = -20; u += 50''', method='euler')
neuron_groups['GC'].I = 0
neuron_groups['GC'].v = -50
neuron_groups['GC'].vr = -50
neuron_groups['GC'].vt = -20
neuron_groups['GC'].k1 = 0.058
neuron_groups['GC'].k2 = 0.625
neuron_groups['GC'].C = 7.1
neuron_groups['GC'].a = 0.0167
neuron_groups['GC'].b = -0.94
neuron_groups['GC'].tau = 20*ms

# define STDP connections
tc_pre_ee = 20*b2.ms
tc_post_1_ee = 20*b2.ms
tc_post_2_ee = 40*b2.ms
nu_ee_pre =  0.1      # ee learning rate
nu_ee_post = 0.3       # ee learning rate
nu_ei_pre =  6.25      # ei learning rate
nu_ei_post = 62.5       # ei learning rate
nu_ie_pre =  12.5      # ie learning rate
nu_ie_post = 125       # ie learning rate
wmax_ee = 1000
wmax_ei = 1500
wmax_ie = 2300
eqs_stdp = '''
                w : 1
                post2before                            : 1
                dpre/dt   =   -pre/(tc_pre_ee)         : 1 (clock-driven)
                dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (clock-driven)
                dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (clock-driven)
             '''
eqs_stdp_pre_ee = '''
             I_post += w
             pre = 1.
             w = clip(w + nu_ee_pre * post1, -wmax_ee, wmax_ee)
             '''
eqs_stdp_post_ee = '''
             post2before = post2
             w = clip(w + nu_ee_post * pre * post2before, -wmax_ee, wmax_ee)
             post1 = 1.; post2 = 1.
             '''
eqs_stdp_pre_ei = '''
             I_post += w
             pre = 1.
             w = clip(w + nu_ei_pre * post1, 0, wmax_ei)
             '''
eqs_stdp_post_ei = '''
             post2before = post2
             w = clip(w + nu_ei_post * pre * post2before, 0, wmax_ei)
             post1 = 1.; post2 = 1.
             '''
eqs_stdp_pre_ie = '''
             I_post += w
             pre = 1.
             w = clip(w - nu_ie_pre * post1, -wmax_ie, 0)
             '''
eqs_stdp_post_ie = '''
             post2before = post2
             w = clip(w - nu_ie_post * pre * post2before, -wmax_ie, 0)
             post1 = 1.; post2 = 1.
             '''

# input -> ORN
connections['S0'] = Synapses(input_groups['input_neurons'], neuron_groups['ORN'], model='''w : 1''',
                             on_pre='''I_post += w''', method='exact')
connections['S0'].connect(j='i')
connections['S0'].w = 3750

# ORN -> MC
connections['S1'] = Synapses(neuron_groups['ORN'], neuron_groups['MC'], model=eqs_stdp, on_pre=eqs_stdp_pre_ee,
                             on_post=eqs_stdp_post_ee, method='exact')
connections['S1'].connect(j='i')
connections['S1'].w = 300

# MC -> GC
connections['S2'] = Synapses(neuron_groups['MC'], neuron_groups['GC'], model=eqs_stdp, on_pre=eqs_stdp_pre_ei,
                             on_post=eqs_stdp_post_ei, method='exact')
connections['S2'].connect(j='i')
connections['S2'].w = 600

# GC -> MC
connections['S3'] = Synapses(neuron_groups['GC'], neuron_groups['MC'], model=eqs_stdp, on_pre=eqs_stdp_pre_ie,
                             on_post=eqs_stdp_post_ie, method='exact')
connections['S3'].connect(condition='i!=j')
connections['S3'].w = -800
# visualise_connectivity(connections['S3'])

state_monitors['M0'] = StateMonitor(neuron_groups['ORN'], 'v', record=True)
spike_monitors['ORN'] = SpikeMonitor(neuron_groups['ORN'])
state_monitors['M1'] = StateMonitor(neuron_groups['MC'], 'v', record=True)
spike_monitors['MC'] = SpikeMonitor(neuron_groups['MC'], variables='v')
state_monitors['M2'] = StateMonitor(neuron_groups['GC'], 'v', record=True)
spike_monitors['GC'] = SpikeMonitor(neuron_groups['GC'])
state_monitors['M_S1'] = StateMonitor(connections['S1'], ['w', 'pre', 'post1', 'post2'], record=True)
state_monitors['M_S2'] = StateMonitor(connections['S2'], ['w', 'pre', 'post1', 'post2'], record=True)
state_monitors['M_S3'] = StateMonitor(connections['S3'], ['w', 'pre', 'post1', 'post2'], record=True)
M_SN = [state_monitors['M_S1'], state_monitors['M_S2'], state_monitors['M_S3']]

#------------------------------------------------------------------------------
# run the simulation and set inputs
#------------------------------------------------------------------------------

net = Network()
for obj_list in [input_groups, neuron_groups, connections, state_monitors, spike_monitors]:
    for key in obj_list:
        net.add(obj_list[key])

# 先让net跑0秒是为了消除spikegeneratorgroup在0秒处的脉冲影响
net.run(0*second)

num_examples = 0
for neuron_indices, neuron_spike_times in zip(sensor_indices_copy_list, input_spike_times_list):
    input_groups['input_neurons'].set_spikes(indices=neuron_indices, times=neuron_spike_times + num_examples * run_time)
    net.run(run_time)
    print("----------the network has trained %s samples----------" % (num_examples + 1))
    num_examples += 1

net.stop()

#------------------------------------------------------------------------------
# save weight results
#------------------------------------------------------------------------------

print('save weight results...')
starting = 'breath_'
ending = '_3'
for connName in connections:
    print('saving ' + connName + '...')
    conn = connections[connName]
    # 将zip数据封装为list(zip())
    connListSparse = list(zip(conn.i, conn.j, conn.w))
    np.save('./weights/' + starting + connName + ending, connListSparse)
print('saving weights finished')

#------------------------------------------------------------------------------
# plot results
#------------------------------------------------------------------------------

# 查看ORN、MC和GC中的各个神经元的脉冲发放情况
figure(figsize=(6, 6))
for i, name in enumerate(spike_monitors):
    # i : 0, 1, 2; name : ORN, MC, GC
    subplot(len(spike_monitors), 1, 1+i)
    plot(spike_monitors[name].t/(1000 * ms), spike_monitors[name].i, '.')
    title('Spikes of population ' + name)
subplots_adjust(wspace=0.5, hspace=0.5)

figure(figsize=(12, 6))
num = 1
# 查看脉冲发放频率最高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        subplot(2, 3, num)
        plot(M_Sn.t / (1000 * ms), M_Sn.w[0], label='w, ie 0-1')
        legend(loc='best')
        xlabel('Time (s)')
        title('Synapse_%s' % num)
    else:
        subplot(2, 3, num)
        plot(M_Sn.t / (1000 * ms), M_Sn.w[0], label='w, ee 0-0')
        legend(loc='best')
        xlabel('Time (s)')
        title('Synapse_%s' % num)
    num += 1

num = 1
# 查看脉冲发放频率较高的神经元的三组权重的变化
for M_Sn in M_SN:
    if M_Sn is state_monitors['M_S3']:
        subplot(2, 3, 3 + num)
        plot(M_Sn.t / (1000 * ms), M_Sn.w[1], label='w, ie 1-0')
        legend(loc='best')
        xlabel('Time (s)')
        title('Synapse_%s' % num)
    else:
        subplot(2, 3, 3 + num)
        plot(M_Sn.t / (1000 * ms), M_Sn.w[1], label='w, ee 1-1')
        legend(loc='best')
        xlabel('Time (s)')
        title('Synapse_%s' % num)
    num += 1
subplots_adjust(wspace=0.25, hspace=0.5)

show()
