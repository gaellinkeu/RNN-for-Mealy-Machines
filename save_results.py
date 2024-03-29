import os

def save_training_results(results_filepath, id, times, x_train, x_test, acc, test_acc, dev_acc):
    f1 = open(results_filepath, 'a+')
    if os.path.getsize(results_filepath) == 0:
        f1.write('ID,Time,Train_set_size,Test_set_size,training_acc,testing_acc,dev_accuracy')
    
    f1.write(f'\n{id},{times},{x_train.shape[0]},{x_test.shape[0]},{acc*100},{test_acc[1]*100},{dev_acc[1]*100}')
    f1.close()

def save_extraction_results(id, results_filepath, times, data_size,sim_threshold,final_acc,final_dev_acc,initial_train_acc,initial_dev_acc,all_merges,correct_merges,state_set_size,equivalence):
    f1 = open(results_filepath, 'a+')
    if os.path.getsize(results_filepath) == 0:
        f1.write('ID,Time,data_size,sim_threshold,Final_acc,Final_dev_acc,initial_train_acc,initial_dev_acc,all_merges,correct_merges,state_set_size,equivalence')
    
    f1.write(f'\n{id},{times},{data_size},{sim_threshold},{final_acc},{final_dev_acc},{initial_train_acc},{initial_dev_acc},{all_merges},{correct_merges},{state_set_size},{equivalence}')
    f1.close()

def save_final_report(filepath, id, times, sim_threshold, equivalence, all_merges, correct_merges, n_expected_fsm_states, n_merged_fsm_states, n_dev_inputs, max_dev_length):
    
    f1 = open(filepath, "a")
    f1.write(f'\n\nThe ID: {id}')
    f1.write(f'\nThe times: {times}')
    f1.write(f'\nConcerning Final FSM')
    f1.write(f'\nThe similarity threshold: {sim_threshold}')
    f1.write(f'\nThe equivalence decision: {equivalence}')
    f1.write(f'\nThe total amount of all merging: {all_merges}')
    f1.write(f'\nThe total amount of correct merging: {correct_merges}')
    f1.write(f'\nThe size of expected states set: {n_expected_fsm_states}')
    f1.write(f'\nThe size of obtained states set: {n_merged_fsm_states}')
    f1.write(f'\nThe size of te dev set: {n_dev_inputs}')
    f1.write(f'\nThe max length of a dev word: {max_dev_length}')
    f1.close()