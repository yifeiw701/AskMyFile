css = '''
<style>
.chat-message {
    padding: 1.2rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #008B8B
}
.chat-message.bot {
    background-color: #66CDAA
}
.chat-message .message {
  padding: 0 1.5rem;
  color: #fff;
  font-size: 17px; 
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>

'''

