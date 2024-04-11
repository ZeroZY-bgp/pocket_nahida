import utils


def create_dpo_data(agent, user_name, question_path=None):
    # 仅保存一条一条问答数据，整合数据由其他脚本完成
    save_folder = 'train/datas/dpo/'
    print("当前记录下，输入c记录‘chosen’数据，输入r记录‘rejected’数据，输入q取消。")
    if question_path:
        all_q_str = utils.load_txt_to_str(question_path)
        questions = all_q_str.split('\n\n')

    q_i = 0

    while True:

        # 初始化数据
        res = {
            'prompt': '',
            'chosen': '',
            'rejected': ''
        }

        # prompt
        if question_path:
            if q_i >= len(questions):
                break
            user_prompt = questions[q_i]
            q_i += 1
        else:
            user_prompt = input("prompt: " + user_name + "：")

        user_prompt = user_name + "：" + user_prompt if user_name != "" else user_prompt
        res['prompt'] = user_prompt

        while True:

            response = agent.chat(user_prompt)
            print(user_prompt)
            print(agent.role_name + "：" + response)
            agent.clear_messages()
            user_command = input("command: ")

            if user_command == 'c':
                res['chosen'] = response
            elif user_command == 'r':
                res['rejected'] = response
            elif user_command == 'q':
                print("取消记录。")
                break
            else:
                pass
            print("当前数据：", res)
            can_save = input("是否保存? y/n")
            if can_save == 'y':
                dpo_history_path = save_folder + user_prompt + '.json'
                utils.save_json(dpo_history_path, res)
                print(f"保存于{dpo_history_path}。")
                break

