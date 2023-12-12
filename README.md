* Getting Started with Your Conversational AI Assistant: Initial Setup Guide:

1. Create conda env:

conda create -n SummerCampAsistantLangChain python=3.10.11 <!--check python version firsst import sys print(sys.version) python3 --version
2. Activate conda env:

conda activate SummerCampAsistantLangChain

3. To set up your environment to run the code, first install all requirements:

pip install -r requirements.txt

4. Set the API Key as an Environment Variable

- On Windows use Terminal command:

set OPENAI_API_KEY=your_api_key_here

- On macOS or Linux, use the Terminal comand:

export OPENAI_API_KEY=your_api_key_here


* Open Questions:

- Q:How would you optimize the process if you had more time?
  A: 1. If I had more time I would use LangChain to streamline the process and reduce costs.
     2. I would use a history table of user prompts and LLM answers (I will detail the idea at the meeting).

- Q:How would you test the prompts' performance?
  A:I will manage a number of conversation scripts that I will create quickly with LLM (GPT).
    To reduce costs I will use shorter "camp version details" text is a shortened version of summer_camp_details parameter.
  
- Q:What edge cases do you think are not handled currently that you would add?
  A:User prompts that may be interpreted either as requests for information or as inquiries about registration processes.
  


