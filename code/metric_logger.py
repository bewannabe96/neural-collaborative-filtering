from torch.utils.tensorboard import SummaryWriter


class MetricLogger:
    def __init__(self, evaluation_k, tensorboard_log_dir=None):
        self.hr_title = 'HR@' + str(evaluation_k)
        self.m_ap_title = 'mAP@' + str(evaluation_k)
        self.n_dcg_title = 'nDCG@' + str(evaluation_k)

        self.tensorboard = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir is not None else None

    def log(self, step, group, hr, m_ap, n_dcg):
        print('STEP={};\t\t{}/{}={:.3f};\t\t{}/{}={:.3f};\t\t{}/{}={:.3f};\t\t'.format(
            step, self.hr_title, group, hr, self.m_ap_title, group, m_ap, self.n_dcg_title, group, n_dcg
        ))

        if self.tensorboard is not None:
            self.tensorboard.add_scalar(self.hr_title + '/' + group, hr, step)
            self.tensorboard.add_scalar(self.m_ap_title + '/' + group, m_ap, step)
            self.tensorboard.add_scalar(self.n_dcg_title + '/' + group, n_dcg, step)

    def log_loss(self, step, group, loss):
        print('STEP={};\t\tLOSS/{}={:.3f};'.format(step, group, loss))

        if self.tensorboard is not None:
            self.tensorboard.add_scalar('LOSS/' + group, loss, step)

    def log_train_time(self, step, train_time):
        if self.tensorboard is not None:
            self.tensorboard.add_scalar("TRAIN_TIME", train_time, step)

