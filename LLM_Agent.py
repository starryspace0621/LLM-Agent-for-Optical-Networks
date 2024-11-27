# 标准库导入
import json
import ast
from typing import Optional, Type, Dict, Any, Union
import os
# from dataclasses import dataclass
from datetime import datetime
import re
import warnings  
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 第三方库导入
import numpy as np
# from sklearn.metrics import mean_squared_error

# LangChain 相关导入
import langchain
from langchain import hub
from langchain.agents import AgentOutputParser, AgentType, initialize_agent, AgentExecutor, create_react_agent
from langchain.chains import LLMMathChain
from langchain.schema import AgentAction, AgentFinish
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain_community.tools import (
    BaseTool,
    StructuredTool,
    Tool,
    tool,
    DuckDuckGoSearchRun
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai import ChatOpenAI
from langchain_experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain_community.utilities import SerpAPIWrapper

# Pydantic 相关导入
from pydantic import BaseModel, Field

# 本地模块导入
import Raman_simulator_for_LLM.scripts.propagation_Raman as raman
from Raman_simulator_for_LLM.gnpy.tools.GlobalControl import GlobalControl
GlobalControl.init_logger('log'+datetime.now().strftime("%Y%m%d-%H%M%S"), 1, 'modified')
logger = GlobalControl.logger
logger.debug('All packages are imported. Logger is initialized.')


# 模型设置
langchain.debug = False
langchain.verbose =False

# 设置代理
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'


class PrintingAgentOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print(f"Agent output: {text}")  # 打印原始输出
        
        # 解析逻辑
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        action_match = re.search(r'Action: (.*?)[\n]', text)
        action_input_match = re.search(r'Action Input: (.*)', text)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            return AgentAction(tool=action, tool_input=action_input, log=text)
        
        raise ValueError(f"Could not parse agent output: {text}")


# raman_simulator 工具定义
class raman_simulator_Input(BaseModel):
    input_str: str = Field(description="A string containing raman_pump_power (list of 6 float values, in Watt) and bool_plot (True/False).")


class custom_raman_simulator(BaseTool):
    name = "raman_simulator"
    description = '''
        Args: 
            input_str: A string containing raman_pump_power (list of 6 float values) and bool_plot (True/False).
        Returns: Fifty float values representing the net gain across the raman amplifier.
        Note: Calculates the gain spectrum based on six Raman pump powers and returns the net gain across the raman amplfier.
        '''
    args_schema: Type[BaseModel] = raman_simulator_Input
    return_direct: bool = False

    def _run(
        self, 
        input_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        try:
            # 移除可能存在的外部引号和多余的空白
            input_str = input_str.strip().strip("'\"")
            
            # 尝试直接分割输入
            try:
                raman_pump_power_str, bool_plot_str = input_str.rsplit(',', 1)
                raman_pump_power = ast.literal_eval(raman_pump_power_str)
                bool_plot = bool_plot_str.strip().lower() == 'true'
            except:
                # 如果直接分割失败，尝试使用 json.loads
                try:
                    parsed_input = json.loads(f"[{input_str}]")
                    raman_pump_power, bool_plot = parsed_input
                except json.JSONDecodeError:
                    raise ValueError("Unable to parse input string")

            # 确保 raman_pump_power 是一个列表
            if not isinstance(raman_pump_power, list) or len(raman_pump_power) != 6:
                raise ValueError("raman_pump_power should be a list of 6 float values")

            # 确保 bool_plot 是一个布尔值
            bool_plot = bool(bool_plot)

            return raman.raman_transmit(raman_pump_power, bool_plot=bool_plot)
        except Exception as e:
            raise ValueError(f"Invalid input format. Expected 'raman_pump_power, bool_plot'. Error: {str(e)}")

    async def _arun(self, 
                    input_str: str, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool asynchronously.'''
        return self._run(input_str, run_manager=run_manager.get_sync())

# net gain 求平均工具定义
def average_all(net_gain):

    """take average of all net gain

    Args: 50 float values representing the net_gains.
    Returns: 10 float values representing the average net gain every 5 value in net_gain

    Note: calculate the average net_gain on the net_gain
    """
    
    return np.mean(np.array(net_gain))
    
class average_all_Input(BaseModel):
    net_gain: list | str = Field(description="a list of 50 input net_gain,in dB")
    
class custom_average_all(BaseTool):
    name = "average_all"
    description = '''
        Args: A list, containing 50 float values representing the net_gains.
        Returns: 1 float value representing the average net gain 

        Note: calculate the averge value of the net gain and then output 1 float net gain
        '''
    args_schema: Type[BaseModel] = average_all_Input
    return_direct: bool = False

    def _run(
        self, 
        net_gain: list | str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->str:
        '''use the tool'''
        # 确保 net_gain 是一个列表
        if isinstance(net_gain, str):
            try:
                net_gain = ast.literal_eval(net_gain)
            except:
                raise ValueError("Invalid input for net_gain. Expected a list.")
        return average_all(net_gain)
    
    async def _arun(self, 
                    net_gain: list | str, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        '''use the tool  asynchronously.'''
        return self._run(ast.literal_eval(net_gain), run_manager=run_manager.get_sync())
    

import NN_raman
# NN_train 工具定义
class NN_train_Input(BaseModel):
    input_str: str = Field(description='''A string containing 'layer_size'(the layer size of the model), 
                           'inputs_file' (the filename of the file to store the inputs data), 'outputs_file' (the filename of the file to store the outputs data), 
                           'activation' (the name of activation function of the model)
                           'criterion'(the name of error function of the model), 'optimizer'(the name of optimizer of the model).''')


class custom_NN_train(BaseTool):
    name = "NN_train_tool"
    description = '''
        Args: 
            input_dict: A string containing 'layer_size'(the layer size of the model), 
                    'inputs_file' (the filename of the file to store the inputs data), 'outputs_file' (the filename of the file to store the outputs data),
                    'activation' (the name of activation function of the model),
                    'criterion'(the name of error function of the model), 'optimizer'(the name of optimizer of the model).
            Returns: a string of the filename which save the trainning parameters. The string is named like 'hidden_size,activation,criterion,optimizer,lossvalue.pth'.
            Note: Training an MLP neural network to predict the relation between raman_pump and net gain.
        '''
    args_schema: Type[BaseModel] = NN_train_Input
    return_direct: bool = False

    def _run(
        self, 
        input_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->dict:
        '''use the tool'''
        # print(input_str)
        input_str = input_str.strip().strip("'\"")
        input_str = input_str.replace(' ', '')
        try:
            hiddens_sizes_str, input_file, output_file, activation, criterion, optimizer = input_str.rsplit(',', 5)
            hiddens_sizes = ast.literal_eval(hiddens_sizes_str)
        except (ValueError, SyntaxError):
            try:
                parsed_input = json.loads(f"[{input_str}]")
                hiddens_sizes, input_file, output_file, activation, criterion, optimizer = parsed_input
                hiddens_sizes = ast.literal_eval(hiddens_sizes)
            except json.JSONDecodeError:
                raise ValueError("Unable to parse input string.")

        if not isinstance(hiddens_sizes, list) or not all(isinstance(i, int) for i in hiddens_sizes):
            raise ValueError("layer_sizes should be a list of integers.")
        

        return NN_raman.NN_train(input_file, output_file, criterion, optimizer, hiddens_sizes, activation)

    async def _arun(self, 
                    input_str: str, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> dict:
        '''use the tool asynchronously.'''
        return self._run(input_str, run_manager=run_manager.get_sync())

# NN_test 工具定义
class NN_test_Input(BaseModel):
    input_str: str = Field(description='''a string of the filename which save the trainning parameters.''')

class custom_NN_test(BaseTool):
    name = "NN_test_tool"
    description = '''
        Args: 
            input_dict: a string of the filename which save the trainning parameters.
            Returns: the test_error of this model.
            Note: Test the model you have trained.
        '''
    args_schema: Type[BaseModel] = NN_test_Input
    return_direct: bool = False

    def _run(
        self, 
        input_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->dict:
        '''use the tool'''
        input_str = input_str.strip().strip("'\"")
        if not input_str[-3:] == 'pth':
            raise FileNotFoundError(f"The file '{input_str}' does not exist.")
        
        return NN_raman.NN_test(input_str)

    async def _arun(self, 
                    input_str: str, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> dict:
        '''use the tool asynchronously.'''
        return self._run(input_str, run_manager=run_manager.get_sync())

# NN_pred 工具定义
class NN_pred_Input(BaseModel):
    input_str: str = Field(description='''a string contain the raman_pump_power (list of 6 float values, in Watt) and the name of the file which save the model's parameters.''')


class custom_NN_pred(BaseTool):
    name = "NN_pred_tool"
    description = '''
        Args: 
            input_dict: a string contain the raman_pump_power (list of 6 float values, in Watt) and the name of the file which save the model's parameters.
            Returns: the prediction of the net_gain based on the raman_pump_power.
            Note: using the model you established to predict the net_gain.
        '''
    args_schema: Type[BaseModel] = NN_pred_Input
    return_direct: bool = False

    def _run(
        self, 
        input_str: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    )->dict:
        '''use the tool'''
        try:
            # 移除可能存在的外部引号和多余的空白
            input_str = input_str.strip().strip("'\"")
            input_str = input_str.replace(' ', '')
            
            # 尝试直接分割输入
            try:
                raman_pump_power = []
                parts = input_str.split(',', 6)
                pth = parts[6]
                for i in range(6):
                    if '[' in parts[i]:
                        parts[i] = parts[i][1:]
                    if ']' in parts[i]:
                        parts[i] = parts[i][:-1]
                    raman_pump_power.append(float(parts[i]))
            except:
                # 如果直接分割失败，尝试使用 json.loads
                try:
                    parsed_input = json.loads(f"[{input_str}]")
                    raman_pump_power, bool_plot = parsed_input
                except json.JSONDecodeError:
                    raise ValueError("Unable to parse input string")

            # 确保 raman_pump_power 是一个列表
            if not isinstance(raman_pump_power, list) or len(raman_pump_power) != 6:
                raise ValueError("raman_pump_power should be a list of 6 float values")

            # 确保 pth 是'pth'文件
            if not pth[-3:] == 'pth':
                raise FileNotFoundError(f"The file '{input_str}' does not exist.")
        except Exception as e:
             raise ValueError(f"Invalid input format. Expected 'raman_pump_power, pth'. Error: {str(e)}")
    
        return NN_raman.NN_pred(raman_pump_power, pth)

    async def _arun(self, 
                    input_str: str, 
                    run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> dict:
        '''use the tool asynchronously.'''
        return self._run(input_str, run_manager=run_manager.get_sync())


if __name__ == '__main__':
    raman_simulator=custom_raman_simulator()
    average_all_func=custom_average_all()
    NN_train = custom_NN_train()
    NN_test = custom_NN_test()
    NN_pred = custom_NN_pred()
    # print(NN_pred.run('[0.18, 0.25, 0.17, 0.26, 0.24, 0.13], [64,128],relu,MSELoss,Adam,0.0015.pth'))

    input_template1 = """
    you are a scientist, you should first train a MLP neural network to model the relationship between the raman power and the net gain. The model should be as accurate as possible. The total loss should be less than 0.015. 
    Then you should test the model by calculate the test_error. There is no requirement for the test error.
    At last, you should use the model to predict the net gain when the raman_pump_power is [0.18, 0.25, 0.17, 0.26, 0.24, 0.13].
    You cannot calculate by yourself but get information from the tools.
    additional information for you is:
    1. tools: 
        NN_train(input_str): 
            - input_str should be a string containing inputs_file, outputs_file, criterion, optimizer, separated by a comma.
            - inputs_file should be the name of the file which stores the raman pump value(inputs). in this case, it should be 'raman_pump.json'
            - outputs_file should be the name of the file which stores the net gain(outputs). in this case, it should be 'net_gain.json'
            - criterion should be the loss function of the model, eg. L1Loss, MSELoss, SmoothL1Loss. You can choose MSELoss first than try others to find the fittest one.
            - optimizer should be the optimizer of the model, eg. SGD, Adam, RMSprop, ASGD, LBFGS. You can choose Adam first than try others to find the fittest one.
            - you can use this tool to train the network. It will return name of the file which store the parameters of the network. the name of the file is like 'criterion_optimizer_lossvalue.pth'.
            - you can obtain the loss_value in the filename. 
            - Important: When calling NN_train, make sure to use the correct format. 
                The input should be a string containing four parts separated by a comma.
                Input example: 'inputs.json, outputs.json, MSELoss, SGD'
                Do not add any extra quotes or characters to this input.   
        NN_test(input_str):
            - input_str should be a string of the filename which save the trainning parameters.
            - you can use this tool to train the network. It will return the test_error of the model.
            - Important: When calling NN_test, make sure to use the correct format. 
                The input should be a string ending in 'pth'.
        NN_pred(input_str):
            - input_str should be a string contain the raman_pump_power (list of 6 float values, in Watt) and the name of the file which save the model's parameters.
            - you can use this tool to predict the net_gain based on the model you established.
            - Important: When calling NN_train, make sure to use the correct format. 
                The input should be a string containing four parts separated by a comma.
                Input example: '[0.04, 0.03, 0.02, 0.12, 0.03, 0.03], MSELoss_Adam_0.0007.pth'

    4. output the filename, the test_error and the prediction.
    5. suggestion: you can first try MSELoss and Adam
    6. suggestion: you can obtain the total loss of the network in the filename
    """



    input_template2 = """
    you are a scientist, you should optimize and adjust the raman power to make the average net gain approach -4dB as close as possible. The largest acceptable error is 0.01dB.
    Plots are needed.
    You cannot calculate by yourself but get information from the tools.
    additional information for you is:
    1. tools: 
        raman_simulator(input_str): 
            - input_str should be a JSON-formatted string containing raman_pump_power and bool_plot., separated by a comma.
            - raman_pump_power should be a list of 6 float values, e.g., [0.04, 0.03, 0.02, 0.12, 0.03, 0.03]
            - bool_plot should be True or False to indicate whether to generate a plot
        you can use this tool to calculate the net gain of specific raman power. It will return the net gain.
        Important: When calling raman_simulator, make sure to use the correct format. 
            The input should be a string containing two parts separated by a comma.
            Input example: '[0.04, 0.03, 0.02, 0.12, 0.03, 0.03], true'
            Do not add any extra quotes or characters to this input.    
        
        average_all_func(net_gain): 
            - net_gain should be a list of float values
        you can use this tool to calcualte the average value of the net gain with the tool. It will return the aeverage value of net gain.
        raman_reverse_simulator(input_str):
            - net gain should be a list of 50 float values
        you can use this tool to calculate the raman_pump_power based on the net gain. It will return the raman pump power.
    2. limitation: the adjustable range of each Raman power is 0 to 0.3. The value of each raman power can be different.
    3. requirement: you should use the least iteration of calculation to find the most suitable raman power.  
    4. output the final net gain and the raman power.
    5. suggestion: you can start by using the raman_reverse_simulator tool to predict the approximate range of pumping power
    5. suggestion: you can first calculate the Raman power of [0.04, 0.03, 0.02, 0.12, 0.03, 0.03], [0.14, 0.11, 0.08, 0.05, 0.01, 0.01],[0.1, 0.03, 0.12, 0.02, 0.1, 0.06] as examples
    """

    input_template3 = """
    you are a scientist, you should first train a NN neural network to model the relationship between the raman power and the net gain. The model should be as accurate as possible. The total loss should be less than 0.015. 
    Then you should test the model by calculate the test_error. There is no requirement for the test error.
    At last, you should use the model to predict the net gain when the raman_pump_power is [0.18, 0.25, 0.17, 0.26, 0.24, 0.13].
    You cannot calculate by yourself but get information from the tools.
    additional information for you is:
    1. tools: 
        NN_train(input_str): 
            - input_str should be a string containing hiddens_size, inputs_file, outputs_file, activation, criterion, optimizer, separated by a comma.
            - hiddens_size should be list of integers, representing the number of neurons in each hidden layer
            - inputs_file should be the name of the file which stores the raman pump value(inputs). in this case, it should be 'raman_pump.json'
            - outputs_file should be the name of the file which stores the net gain(outputs). in this case, it should be 'net_gain.json'
            - activation: String, specifies the activation function (e.g., 'relu', 'tanh', 'sigmoid'). You can choose the one you think is best first.
            - criterion: String, specifies the criterion function, which should be from the PyTorch library. You can choose the one you think is best first.
            - optimizer should be the optimizer of the model, which should be from the PyTorch library. You can choose the one you think is best first.
            - you can use this tool to train the network. It will return name of the file which store the parameters of the network. the name of the file is like 'hidden_size,activation,criterion,optimizer,lossvalue.pth'.
            - you can obtain the loss_value in the filename. 
            - Important: When calling NN_train, make sure to use the correct format. 
                The input should be a string containing six parts separated by a comma.
                Input example: '[64, 128], inputs.json, outputs.json, activation, criterion, optimizer'
                Do not add any extra quotes or characters to this input.   
        NN_test(input_str):
            - input_str should be a string of the filename which save the trainning parameters.
            - you can use this tool to train the network. It will return the test_error of the model.
            - Important: When calling NN_test, make sure to use the correct format. 
                The input should be a string ending in 'pth'.
        NN_pred(input_str):
            - input_str should be a string contain the raman_pump_power (list of 6 float values, in Watt) and the name of the file which save the model's parameters.
            - you can use this tool to predict the net_gain based on the model you established.
            - Important: When calling NN_train, make sure to use the correct format. 
                The input should be a string containing four parts separated by a comma.
                Input example: '[0.04, 0.03, 0.02, 0.12, 0.03, 0.03], MSELoss_Adam_0.0007.pth'

    4. output the filename, the test_error and the prediction.
    5. suggestion: you can obtain the total loss of the network in the filename
    """

    template = {
        "input": input_template3,
        "intermediate_steps": []
    }

    llm = ChatOpenAI(openai_api_key = "YOUT API HERE", model="gpt-4o-mini", temperature=0)
    tools = [NN_train, NN_test, NN_pred]
    prompt = hub.pull("hwchase17/react")
    output_parser = PrintingAgentOutputParser()
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=100000, handle_parsing_errors=True)

    try:
        result = agent_executor.invoke({"input": template})
        # logger.info(f"Agent execution result: {result}")
    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}")
        import traceback
        logger.error(traceback.format_exc())


    print('Hello, world!')
