import torch

class option():


    # main encoder decoder option


    n_layers=6
    n_position = 160

    # state tracker & policy network option
    n_inner_vocab = 5000
    n_inner_layers = 3
    n_inner_position = 15

    # shared encoder decoder option
    d_word_vec = 512
    n_head = 8
    d_k = 64
    d_v = 64
    pad_idx = 2
    d_model = 512
    d_inner = 2048
    dropout = 0.1
    n_warmup_steps = 2000
    scale_emb = False
    switch_interval = 16  # 4 times $gradient_stack$

    # *** data configuration ***
    cache_turn = 0
    intention_dict = {"c": 0,  # chitchat
                      "a": 0,  # apprentice
                      "s": 1,  # ask symptom
                      "d": 2,  # diagnosis
                      "m": 3,  # medicine
                      "t": 3}  # treatment
    fullname2abbr = {
        "chitchat": "c",
        "apprentice_response": "a",
        "prescribe_medicine": "m",
        "ask_symptoms": "s",
        "diagnosis_disease": "d",
        "diagnosis_treatment": "t"
    }
    # remove the unused turn
    discourse_type = "a,s,d,m,t".split(",")
    discourse_type_set = set(discourse_type)

    # *** input data & output data configuration***
    context_max_len = 200
    r_max_len = 50
    r_beam_max_len = 30
    conv_max_len = 500
    profile_num = 10
    state_num = 10
    state_num_redial = 20
    pretrain_state_num = 50
    all_topic_num = 20
    all_topic_num_redial = 40
    movie_path_len = 3
    tag_num = 3
    preference_num = 5
    topic_num = 2
    action_num = 10
    action_num_redial = 1
    worker_num = 1
    relation_num = 150
    movie_num = 200
    state_token = 40

    scale_prj = True
    # seq_logit *= self.d_model ** -0.5
    epoch = 100
    # if VO.scale_prj:
    #     logits *= VO.d_model ** -0.5

    # for input filename
    task = "meddg"
    # dialogue sessions
    dataset_file = "dataset/{dataset}.zip"
    # topic
    topic_file = "./dataset/topic_only.txt"
    movie_file = "./dataset/movie_only.txt"
    topic_redial = "./dataset/topic_redial_dbpedia.txt"
    topic_movie_file = "./dataset/topics_allmovie.txt"
    # topic_file = r"C:\Users\Administrator\Desktop\res\topics.txt"
    # profile
    profile_file="./dataset/user2TopicSent.pkl"
    # movie file

    # vocab
    vocab_file="./dataset/vocab_1.txt"
    vocab_movie_file = "./dataset/vocab_3.txt"
    vocab_redial = "./dataset/vocab_redial_dbpedia.txt"
    # vocab_file = r"C:\Users\Administrator\Desktop\res\vocab.txt"
    # special words
    special_words_file = './dataset/topics.txt'
    #  special_words_file = r"C:\Users\Administrator\Desktop\res\topics.txt"
    # for output filename
    test_filename_template = "data/cache/{model}/{uuid}/{epoch}-{global_step}-{mode}-{metric}.txt"
    ckpt_filename_template = "data/ckpt/{model}/{uuid}/{epoch}-{global_step}-{metric}.model.ckpt"

    vocab_size = 30000
    inner_vocab_size = 10000

    #@staticmethod
    # def update_device(device_id):
    #     if device_id < 0:
    #         return
    #     else:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(device_id)
    #         VRBotOption.device = torch.device("cuda:{}".format(0))

    # ablation
    no_action_super = None
    max_patience = 20
    log_loss_interval = 100
    gradient_stack = 8
    decay_interval = 10000
    decay_rate = 0.9
    lr = 5e-5
    mini_lr = 1e-5
    valid_eval_interval = 10000
    test_eval_interval = 10000
    force_ckpt_dump = True
    sub_gen_lambda = 0.01

    s_copy_lambda = 1
    a_copy_lambda = 1
    copy_lambda_mini = 0.1
    copy_lambda_decay_steps = 10000
    copy_lambda_decay_value = 1.0

    init_tau = 1.0
    tau_mini = 0.1
    tau_decay_total_steps = 5000
    tau_decay_rate = 0.5
    beam_width = 5

    # Ablation Study
    wo_l = False
    wo_m = False
    wo_entropy_restrain = False
    wo_repeat_penalty = False
    wo_rl = False
    super_only = False
    hungary = False
    super_rate = 0.0
    super_epoch = 5

    batch_size = 16
    reg_lambda = 5e-3

    # special token

    BOS_CONTEXT = "[s_context]"
    EOS_CONTEXT = "[/s_context]"

    BOS_RESPONSE = "[s_response>]"
    EOS_RESPONSE = "[/s_response]"

    BOS_ACTION = "[s_action]"
    EOS_ACTION = "[/s_action]"

    PAD_WORD = "[PAD]"
    SENTENCE_SPLITER = "[sent]"
    TOPIC_SPLITER = "[unused2]"
    UNK_WORD = "[UNK]"

    BOS_PRE = "[s_preference]"
    EOS_PRE = "[/s_preference]"

    BOS_PRO = "[s_profile]"
    EOS_PRO = "[/s_profile]"