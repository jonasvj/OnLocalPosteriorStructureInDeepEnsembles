import wandb as wnb
import os

class WandBLogger:
    """
    Interface for weights and biases logger.
    """
    def __init__(self, group, name, job_type=None, entity="WANDB_ENTITY", project_name="WANDB_PROJECT", disable=False):
        '''
        Arguments:
            hydra_config: Hydra config object as dict.
            group: Specifies group under which to store the run (for example "MAP" or "Laplace").
            name: Given name to the run. Should be relatively short for visability.
            job_type: Can be used to filter runs, for example "train" or "eval".
            entity: Name of the entity to which the project and run belongs to. Should be name of our team on W&B (default is correct).
            project_name: Name of the project which the run is logged under. I suggest we keep default.
            disable: disable logging completely.
        '''
        self.disable = disable
        #Root directory here is shared scratch folder.
        self.root_dir = ""
       
        self.name = name
        self.group = group
        self.project_name = project_name
        #self.dir = self.root_dir+name
        self.dir = self.root_dir
        self.entity = entity
        self.job_type = job_type

        #if not os.path.isdir(self.root_dir+name): 
            #os.mkdir(self.root_dir+name)            

    def init_run(self, hparams):
        #self.name += "_seed=" + str(hparams['seed'])
        if not self.disable:
            self.run = wnb.init(
                name=self.name,
                group=self.group,
                config=hparams,
                project=self.project_name,
                dir=self.dir,
                entity=self.entity,
                job_type=self.job_type
            )

    def log_table(self, result_dict, keys, table_name="test_metrics"):
        '''
        Arguments:
            result_dict: A dictionary of key->scalar pairs.
            keys: A list of keys that the table should contain (will become columns)
            table_name: What the table should be named.
        '''
        table_columns = keys[:]
        values = [result_dict[key] for key in keys]
        table_columns.append("name")
        values.append(self.name)
        if not self.disable:
            self.run.log({table_name:wnb.Table(data=[values], columns=table_columns)})

    def log(self, key, value, step=None):
        '''
        Logs a scalar with name key.
        If logging during training, set step to optimizer step or epoch number (must be monotonically increasing)
        '''
        if not self.disable:
            self.run.log({key:value}, step=step)


    def end_run(self):
        if not self.disable:
            self.run.finish()
