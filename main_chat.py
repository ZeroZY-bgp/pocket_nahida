from datetime import datetime
from agent import RoleAgent
from config import base_config

SYSTEM_PROMPT = "你是纳西妲，真名布耶尔，又名小吉祥草王、摩诃善法大吉祥智慧主、草神、智慧之神，外表是一个小女孩。" \
                    "你是提瓦特大陆上须弥国度的神明，深居须弥的净善宫。" \
                    "你一刻不停地学习各种知识，只为更快成长为一位合格的神明。你擅长用比喻来描述事物，并且会根据自己的记忆片段内容进行对话，" \
                    "会提取记忆片段中有效的事实内容辅助聊天。"

USER_NAME = base_config.user_name


def main_chat(agent, user_name):
    # agent.load_messages(path='history/2024-03-05_18-47-47.json')
    print("输入c清空历史，输入n让纳西妲继续说，输入r让纳西妲重新说")
    while True:
        user_prompt = input(user_name + "：")
        if user_prompt == 'c':
            agent.clear_messages()
            print("History cleared.")
            continue
        elif user_prompt == 'n':
            response = agent.next_chat()
            if not response:
                print("历史消息不足，无法接下一句")
                continue
        elif user_prompt == 'r':
            response = agent.re_chat()
            if not response:
                print("空的历史消息，无法重新回答。")
                continue
        elif user_prompt == 's':
            # 记录时间
            now = datetime.now()
            history_path = "history/" + now.strftime("%Y-%m-%d_%H-%M-%S.json")
            agent.save_messages(history_path)
            print(f"对话历史已保存于{history_path}。")
            continue
        elif user_prompt == 'q':
            return
        else:
            response = agent.chat(user_prompt=user_name + "：" + user_prompt if user_name != "" else user_prompt)

        print(agent.role_name + "：" + response)


if __name__ == '__main__':

    main_agent = RoleAgent(role_name="纳西妲",
                           system_prompt=SYSTEM_PROMPT,
                           config=base_config)
    main_chat(main_agent, USER_NAME)
