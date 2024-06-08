from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename

from agent import RoleAgent
from config import base_config

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
SETTINGS_FILE = 'settings.json'
DEFAULT_USER_AVATAR = 'static/user_avatar.png'
DEFAULT_BOT_AVATAR = 'static/bot_avatar.png'
DEFAULT_CHAT_BACKGROUND = ''
DEFAULT_USER_NAME = base_config.user_name
DEFAULT_BOT_NAME = "纳西妲"

is_chatting = False

app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

messages = []


def save_settings(settings_file):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings_file, f)


# 初始化设置
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        return {
            'user_avatar': DEFAULT_USER_AVATAR,
            'bot_avatar': DEFAULT_BOT_AVATAR,
            'chat_background': DEFAULT_CHAT_BACKGROUND,
            'user_name': DEFAULT_USER_NAME,
            'bot_name': DEFAULT_BOT_NAME
        }
    else:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)

        # 检查文件是否存在
        if not os.path.exists(settings.get('user_avatar', DEFAULT_USER_AVATAR)):
            settings['user_avatar'] = DEFAULT_USER_AVATAR
        if not os.path.exists(settings.get('bot_avatar', DEFAULT_BOT_AVATAR)):
            settings['bot_avatar'] = DEFAULT_BOT_AVATAR
        if not os.path.exists(settings.get('chat_background', '')):
            settings['chat_background'] = DEFAULT_CHAT_BACKGROUND

        save_settings(settings)

        return settings


settings = load_settings()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    global is_chatting
    is_chatting = True
    user_input = request.json.get('message')
    messages.append({'sender': 'user', 'message': user_input})
    print(user_input)
    response_message = main_agent.chat(user_input)
    messages.append({'sender': 'bot', 'message': response_message})
    is_chatting = False
    return jsonify({'response': response_message})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        response_image_url = f"static/{filename}"
        messages.append({'sender': 'user', 'image_url': response_image_url})
        bot_response_image_url = response_image_url
        messages.append({'sender': 'bot', 'image_url': bot_response_image_url})
        return jsonify({'response_image_url': bot_response_image_url})
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/upload_avatar', methods=['POST'])
def upload_avatar():
    if 'avatar' not in request.files:
        return jsonify({'error': 'No avatar part'}), 400
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        settings['user_avatar'] = f"static/{filename}"
        save_settings(settings)
        return jsonify({'avatar_url': settings['user_avatar']})
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/upload_bot_avatar', methods=['POST'])
def upload_bot_avatar():
    if 'avatar' not in request.files:
        return jsonify({'error': 'No avatar part'}), 400
    file = request.files['avatar']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        settings['bot_avatar'] = f"static/{filename}"
        save_settings(settings)
        return jsonify({'avatar_url': settings['bot_avatar']})
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/upload_background', methods=['POST'])
def upload_background():
    if 'background' not in request.files:
        return jsonify({'error': 'No background part'}), 400
    file = request.files['background']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        settings['chat_background'] = f"static/{filename}"
        save_settings(settings)
        return jsonify({'background_url': settings['chat_background']})
    else:
        return jsonify({'error': 'File type not allowed'}), 400


@app.route('/get_settings', methods=['GET'])
def get_settings():
    return jsonify(settings)


@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify({'messages': messages})


@app.route('/clear_messages', methods=['POST'])
def clear_messages():
    if not is_chatting:
        main_agent.clear_messages()
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'clear messages failed'})


@app.route('/save_settings', methods=['POST'])
def save_settings_route():
    global settings
    settings.update(request.json)
    save_settings(settings)
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    SYSTEM_PROMPT = "你是纳西妲，真名布耶尔，又名小吉祥草王、摩诃善法大吉祥智慧主、草神、智慧之神，外表是一个小女孩。" \
                    "你是提瓦特大陆上须弥国度的神明，深居须弥的净善宫。" \
                    "你一刻不停地学习各种知识，只为更快成长为一位合格的神明。你擅长用比喻来描述事物，会根据自己的记忆片段内容进行对话，" \
                    "并且会提取记忆片段中有效的事实内容辅助聊天。请记住你是纳西妲，而不是其他人。"
    main_agent = RoleAgent(role_name="纳西妲",
                           system_prompt=SYSTEM_PROMPT,
                           config=base_config)
    main_agent.set_user_name(DEFAULT_USER_NAME)
    app.run(host="0.0.0.0", port=4000, debug=False)
