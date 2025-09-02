import model
from flask import Flask,render_template,request
app = Flask(__name__,template_folder='templates')


@app.route('/')
def index():
    return render_template('meww.html')

def result(home,away):
    my_dic={'home_team':home, 'away_team':away,'winning_team': None}
    predict = model.predictwin(my_dic)
    print(predict['match'],'\n',predict['winner'],'\n',predict['pwin_home'],'\n',predict['pdraw'],'\n',predict['pwin_away'])
    return  (predict['match'] , predict['winner'], predict['pwin_home'] ,  predict['pdraw'], predict['pwin_away'])


@app.route('/1', methods =["GET", "POST"])
def change(*args):
    if request.method == "POST":
        global home,away
        #home =  request.form.get("home")
        #away = request.form.get("away")
        home=request.form['HOME']
        away=request.form['AWAY']
        if (home==away):
            return render_template("error404.html")
        else:
            meow=result(home,away)
            #return ("Home Team:"+home +"Away Team:"+away)
            return render_template("meow.html",home=home,away=away,meow=meow)
            # return ("Home team is:"+home  + "Away team is:"+away +  "Result"+meow)
            # return render_template("meww.html")
    return render_template("meow.html")





if __name__ == '__main__':
    app.run(debug=True)