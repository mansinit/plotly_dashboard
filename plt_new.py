from flask import Flask,render_template
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as plt
import json
import plotly

import os
import random
import pandas as pd
from flask import Flask, render_template , request
import sqlite3
from skimage import io
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import matplotlib
PEOPLE_FOLDER = os.path.join('static')
import re
from wordcloud import WordCloud, STOPWORDS
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
tl=[]
data = pd.read_excel('C:\\Users\\Mansi Dhingra\\Desktop\\Projects\\api\\Tweet_qlik_first.xlsx')
from sqlalchemy import create_engine
engine = create_engine('sqlite://', echo=False)


def create_graphs(x,y,string):


    if string=="trigram":
        trace1 = go.Bar(x=x, y=y, marker_color='green')
        layout = go.Layout(title="Top 20 " + "bigram" + " words", xaxis=dict(title="Words", ),
                           yaxis=dict(title="Count", ), autosize=False, width=450, height=350)
    elif string=="positive":

        trace1 = go.Bar(x=x, y=y, marker_color='orange')
        layout = go.Layout(title="Top 10 most "+string+" words", xaxis=dict(title="Words", ),
                       yaxis=dict(title="Count", ), autosize=False, width=450, height=350)
    else:

        trace1 = go.Bar(x=x, y=y, marker_color='red')
        layout = go.Layout(title="Top 10 most "+string+" words", xaxis=dict(title="Words", ),
                       yaxis=dict(title="Count", ), autosize=False, width=450, height=350)
    data = [trace1]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(tickangle=45)
    fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json


def get_top_n_bigram(corpus, n=None):
    stopwords = set(STOPWORDS)
    stopwords.add("amp")
    stopwords.add("please")
    stopwords.add("stop")
    stopwords.add("already")
    stopwords.add("deliver")
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def ret(out):
    list=[]
    for c in range(0, len(out.cat.categories.left)):
        dict = {}
        dict[0] = (out.cat.categories.left[c])
        if dict[0] == 1000:
            dict[0] = "1K"
        if dict[0] == 10000:
            dict[0] = "10K"
        if dict[0] == 50000:
            dict[0] = "50K"
        if dict[0] == 100000:
            dict[0] = "100K"
        if dict[0] == 500000:
            dict[0] = "500K"

        dict[1] = (out.cat.categories.right[c])
        if dict[1] == 1000:
            dict[1] = "1K"
        if dict[1] == 10000:
            dict[1] = "10K"
        if dict[1] == 50000:
            dict[1] = "50K"
        if dict[1] == 100000:
            dict[1] = "100K"
        if dict[1] == 500000:
            dict[1] = "500K"
        if dict[1] == 1000000000:
            dict[1] = "above"
        list.append(dict)
    return list

@app.route('/', methods=['GET', 'POST'])
def homepage():
    if request.method=='GET':

        folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/plotly_dashboard/static/images'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            os.remove(file_path)
        data = pd.read_excel('C:\\Users\\Mansi Dhingra\\Desktop\\Projects\\api\\Tweet_qlik_first.xlsx')
        data.to_sql('users', con=engine)
        search=request.form.get("search")
        topic_list=engine.execute("SELECT distinct Topic FROM users").fetchall()

        count=engine.execute('''Select Topic,count(*) from users group by Topic''',).fetchall()

        tl = []
        for tr in topic_list:
            tl.append(tr[0])
        x = []
        y = []
        list=[]

        for tr in count:
            dict1 = {}
            x.append(tr[0])
            dict1['topic']=tr[0]
            y.append(tr[1])
            dict1['count']=tr[1]
            list.append(dict1)

        newlist = sorted(list, key=lambda k: k['count'], reverse=True)
        newlist=(newlist[1:6])
        toplist = []
        clist = []
        for top in newlist:
            toplist.append(top['topic'])
            clist.append(top['count'])

        trace1 = go.Bar(x=x, y=y)
        layout = go.Layout(title="Overall Tweet Count based on the topics", xaxis=dict(title="Topics",),
                           yaxis=dict(title="Tweet Count",),autosize=False ,width=450,height=350)
        data = [trace1]
        fig = go.Figure(data=data, layout=layout)
        fig.update_xaxes(tickangle=45)
        fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        sent_count = engine.execute('''Select Sentiment,Count(*) from users group by Sentiment''').fetchall()
        sent_topic = []
        sent_count1 = []
        for tx in sent_count:
            sent_topic.append(tx[0])
            sent_count1.append(tx[1])
        piel=go.Pie(labels=sent_topic,
               values=sent_count1,
               hoverinfo='label+value+percent'
               )
        data=[piel]
        layout=go.Layout(title="Sentiment Counts",autosize=False,width=450,height=350)
        fig_pie = go.Figure(data=data, layout=layout)
        fig_json_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)



        fcount = engine.execute('''Select "Follower Count" from users ''').fetchall()
        fcount_list = []
        for tr in fcount:
            if tr[0]:
                fcount_list.append(tr[0])

        toplist = []
        clist = []
        for top in newlist:
            toplist.append(top['topic'])
            clist.append(top['count'])
        sent_full_list=[]
        for tick in toplist:
            sent_list = []
            for sent in sent_topic:
                sentiment = engine.execute('''Select Count(*) from users where Topic=? and Sentiment=?''',
                                           (str(tick), str(sent))).fetchall()
                sent_list.append(sentiment[0][0])
            sent_full_list.append(sent_list)
        tl = []
        for tr in topic_list:
            tl.append(tr[0])


        return render_template("charts.html",plot=fig_json,plot_pie=fig_json_pie,
                               toplist=toplist,clist=clist,sent_full_list=sent_full_list,topics=tl,search=search)

@app.route('/topic_list', methods=['GET', 'POST'])
def topic_list():
    font = {'family': 'sans-serif',
            'sans-serif': 'Ariel'}
    matplotlib.rc('font', **font)
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/plotly_dashboard/static/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    data = pd.read_excel('C:\\Users\\Mansi Dhingra\\Desktop\\Projects\\api\\Tweet_qlik_first.xlsx')
    data.to_sql('users', con=engine)
    topic_list = engine.execute("SELECT distinct Topic FROM users").fetchall()

    count = engine.execute('''Select Topic,count(*) from users group by Topic''', ).fetchall()

    search = request.form.get("search")
    tl = []
    for tr in topic_list:
        tl.append(tr[0])

    result = request.args.get('type')
    print(result)
    '''WORDCLOUD SAVE IT AND DISPLAY IT STATIC'''
    keyword = engine.execute('''Select keywords from users where Topic=?''', (str(result),))
    list_key = []
    images = []
    for x in keyword:
        list_key.append(x[0])

    print(list_key)
    comment_words = ''
    for token in list_key:
        if str(token) != "None":
            comment_words += token
    stopwords = set(STOPWORDS)
    stopwords.add("amp")
    stopwords.add("please")
    stopwords.add("stop")
    stopwords.add("already")
    stopwords.add("economic")
    stopwords.add("country")
    wordcloud = WordCloud(width=1200, height=1200,
                          background_color='white',
                          stopwords=stopwords, max_words=800,
                          min_font_size=10).generate(comment_words)

    # plot the WordCloud image
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    x = random.randint(0, 9999999)
    img_name5 = 'plot_cloud' + str(x) + '.png'
    plt.savefig('static/images/' + img_name5, facecolor="skyblue")

    fcount = engine.execute('''Select "Follower Count" from users where Topic=?''', (str(result),)).fetchall()
    fcount_list = []
    for tr in fcount:
        if tr[0]:
            fcount_list.append(tr[0])

    fcount_series = pd.Series(fcount_list)
    out = pd.cut(fcount_series, bins=[0, 1000, 10000, 50000, 100000, 500000, 1000000000], include_lowest=False)
    val = out.value_counts(sort=False).rename_axis('unique_values').reset_index(name='counts')
    list_val = []
    for each in val['unique_values']:
        list_val.append(each)
    list_val2 = []
    for each in val['counts']:
        list_val2.append(each)

    ax = val.plot.bar(rot=0, figsize=(8, 4), width=0.75, color="red")

    list = ret(out)
    i = 0
    for each in list_val2:
        if i == 0:
            i = -0.1
        else:
            i = i + 1
        if each != 0:
            ax.text(i, each, str(each), fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks([])
    ax.set_xticklabels([(str(c[0]) + " to " + str(c[1])) for c in list], rotation=0)

    ax.set_title('Follower Count for ' + str(result))
    ax.xaxis.labelpad = 20

    x = random.randint(0, 9999999)
    img_name3 = 'plot_fol_count' + str(x) + '.png'
    ax.figure.savefig('static/images/' + img_name3, bbox_inches='tight', pad_inches=0)

    images.append(img_name3)
    images.append(img_name5)

    images_list1 = os.listdir(os.path.join(app.static_folder, "images"))
    tweets = engine.execute('''Select keywords from users where Topic=?''',(str(result),))
    list_tweet = []
    for kw in tweets:
        list_tweet.append(kw[0])
    common_words = get_top_n_bigram(list_key, 20)
    list_words = []
    list_freq = []
    for word, freq in common_words:
        list_words.append(word)
        list_freq.append(freq)
    fig_json_trig = create_graphs(list_words, list_freq, "trigram")

    return render_template("second.html",result=result, img1=images_list1,topics=tl,plot_tri=fig_json_trig,search=search)
@app.route('/search',methods=["get","post"])
def search_func():
    result = request.args.get('search')
    print("*************************************************")
    print(result)  # should print what was the topic that was clicked in the dropdown menu
    font = {'family': 'sans-serif',
            'sans-serif': 'Ariel'}
    matplotlib.rc('font', **font)
    folder = 'C:/Users/Mansi Dhingra/Desktop/Projects/api/plotly_dashboard/static/images'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        os.remove(file_path)
    data = pd.read_excel('C:\\Users\\Mansi Dhingra\\Desktop\\Projects\\api\\Tweet_qlik_first.xlsx')
    data.to_sql('users', con=engine)
    topic_list = engine.execute("SELECT distinct Topic FROM users").fetchall()
    topic_list1=[]
    for tl in topic_list:
        topic_list1.append(tl[0])
    count = engine.execute('''Select Topic,Tweet,"Follower Count" from users where Tweet like ('%' || ? || '%')''',(str(result),) ).fetchall()
    for row in count:
        print(row)
    return render_template("index.html",count=count,topics=topic_list1)
if __name__ == "__main__":
    app.run(debug=True)
