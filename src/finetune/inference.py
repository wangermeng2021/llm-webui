
from abc import ABC, abstractmethod
class Inference():
    @abstractmethod
    def load_model(self):
        pass
    @abstractmethod
    def infer(self):
        pass
    def get_prompt(self,input):
        return self.prompt_template.format(question=input)
    def __call__(self ,input):
        gen_output = self.infer(input)
        return  gen_output
    @abstractmethod
    def free_memory(self):
        pass
