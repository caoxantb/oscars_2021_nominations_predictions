import os
import json
import pandas as pd
from matplotlib import pyplot as plt
import statistics as st
from itertools import chain

class Category:
    def __init__(self, name, awards, extra_col=None, shortlist=None, max_noms=5):
        self.name = name
        self.awards = awards
        self.extra_col = extra_col
        if shortlist == None:
            shortlist = []
        self.shortlist = shortlist
        self.max_noms = max_noms

    def df_to_dict(self, df):
        df.replace(u'\xa0',u'', regex=True, inplace=True)
        sample_dict = {}
        sample_set = set()

        for index, row in df.iterrows():
            year = row["Year"]
            film = row["Film"]
            
            if year not in sample_dict:
                sample_set = set()

            if self.extra_col == "Actor": 
                actor = row["Actor"]
                sample_set.add((actor, film))

            elif self.extra_col == "Song": 
                song = row["Song"]
                sample_set.add((song, film))

            else: 
                sample_set.add(film)
                
            sample_dict[year] = sample_set
        
        return sample_dict

    def df_to_oscar_dict(self):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(THIS_FOLDER, '../dataset/csv/Best {}/Trained Data/oscar.csv'.format(self.name))
        try:
            oscar = self.df_to_dict(pd.read_csv(path, encoding='ISO-8859-1'))
        except:
            oscar = self.df_to_dict(pd.read_csv(path, encoding='utf-8-sig'))

        return oscar

    def read_trained_file(self):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

        awards = []

        for award in self.awards:
            path = os.path.join(THIS_FOLDER, '../dataset/csv/Best {}/Trained Data/{}.csv'.format(self.name, award))

            try:
                award_df = pd.read_csv(path, encoding="ISO-8859-1")
                award_dict = self.df_to_dict(award_df)
            except: 
                award_df = pd.read_csv(path, encoding="utf-8-sig")
                award_dict = self.df_to_dict(award_df)
            finally:
                awards.append(award_dict)
        
        return awards

    def read_applied_file(self):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

        awards = []

        for award in self.awards:
            path = os.path.join(THIS_FOLDER, '../dataset/csv/Best {}/Applied Data/{}.csv'.format(self.name, award))

            try:
                award_df = pd.read_csv(path, encoding="ISO-8859-1")
                award_dict = self.df_to_dict(award_df)
            except: 
                award_df = pd.read_csv(path, encoding="utf-8-sig")
                award_dict = self.df_to_dict(award_df)
            finally:
                awards.append(award_dict[2020])
        
        return awards
    
    def union_all(self):
        oscar = self.df_to_oscar_dict()
        awards = self.read_trained_file()
        union_all = []
    
        for year in range(2010, 2020):
            union_all_each = oscar[year].copy()
            for award in awards:
                union_all_each |= award[year]
            union_all.append(union_all_each)
        
        return union_all

    def calc_bayes_2(self, a, b, n, year, k):
        a_y = a[year]
        b_y = b[year]
        
        a_if_b = len(a_y & b_y)/len(b_y)
        b_if_a = len(a_y & b_y)/len(a_y)
        if k == 0 or (k > 0 and (len(n[year-2010])) > k):
            b_if_not_a = (len(b_y) - len(a_y & b_y))/(len(n[year-2010]) - len(a_y))
        else: 
            b_if_not_a = (len(b_y) - len(a_y & b_y))/(k - len(a_y))

        return (a_if_b, b_if_a, b_if_not_a)
    
    def train_data(self):
        percentage_oscars_if_others_each = []
        percentage_others_if_oscars_each = []
        percentage_others_ifnot_oscars_each = []

        mean_oscars = []    

        oscar = self.df_to_oscar_dict()
        awards = self.read_trained_file()
        union_all = self.union_all()

        for award in awards:
            for year in range(2010, 2020):
                res_1, res_2, res_3 = self.calc_bayes_2(oscar, award, union_all, year, len(self.shortlist))
                percentage_oscars_if_others_each.append(res_1)
                percentage_others_if_oscars_each.append(res_2)
                percentage_others_ifnot_oscars_each.append(res_3)

            mean_oscars.append([st.mean(percentage_oscars_if_others_each), 
                                st.mean(percentage_others_if_oscars_each),
                                st.mean(percentage_others_ifnot_oscars_each)])    

            percentage_oscars_if_others_each = []
            percentage_others_if_oscars_each = []
            percentage_others_ifnot_oscars_each = []
            
        return mean_oscars

    def apply_naive_bayes(self):
        awards_2020 = self.read_applied_file()
        awards_mean = self.train_data()

        if self.shortlist == []:
            awards_2020_union_all = list(set(chain.from_iterable(awards_2020)))
            awards_2020_union_all.append("Others")
        
        else:
            awards_2020_union_all = self.shortlist

        P_OSCARS = self.max_noms/len(awards_2020_union_all)
        P_NOT_OSCARS = 1 - P_OSCARS

        p_final = []

        for film in awards_2020_union_all:
            if self.name == "Original Song":
                film = tuple(film)
            p_calculated = 0
            bayes_n = P_OSCARS
            bayes_d = P_NOT_OSCARS
            
            for i in range(len(awards_2020)):
                if film in awards_2020[i]:
                    bayes_n *= awards_mean[i][1]
                    bayes_d *= awards_mean[i][2]
                
                else:
                    bayes_n *= (1-awards_mean[i][1])
                    bayes_d *= (1-awards_mean[i][2])
            
            p_calculated = bayes_n/(bayes_n + bayes_d)
            p_final.append(p_calculated)
        
        film_final = [i for _, i in sorted(zip(p_final, awards_2020_union_all), reverse=True)]
        p_final = sorted(p_final, reverse=True)
        p_final_100 = list(map(lambda x: x*100, p_final)) 

        return film_final, p_final_100

    def visualize(self):
        film = self.apply_naive_bayes()[0]
        prob = self.apply_naive_bayes()[1]

        if self.extra_col != None:
            film = list(map(lambda name: " - ".join(name) if name != "Others" else name, film))

        fig, ax = plt.subplots(figsize = (16, 9))
        ax.barh(film, prob, color ='gold')

        # Remove axes splines
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)

        # Remove x, y Ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Add padding between axes and labels
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 10)

        # Add x, y gridlines
        ax.grid(b = True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)

        # Show top values 
        ax.invert_yaxis()

        # Add annotation to bars
        for i in ax.patches:
            plt.text(i.get_width()+0.2, i.get_y()+0.5, 
                    " " + str(round((i.get_width()), 2)) + " %",
                    fontsize = 10, fontweight ='bold',
                    color ='grey')

        # Add Plot Title
        ax.set_title('Probability of receiving an Oscar nomination based on other major awards',
                    loc ='center')
        ax.set_ylabel('Films nominated in other major awards', labelpad=30)
        ax.set_xlabel('Probability of nominations (%)', labelpad=30)

        # Add Text watermark
        fig.text(0.9, 0.15, '@caoxantb', fontsize = 12,
                color ='grey', ha ='right', va ='bottom',
                alpha = 0.7)

        # Show Plot
        plt.show()

if __name__ == '__main__':
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    shortlist_json = os.path.join(THIS_FOLDER, '../dataset/json/shortlist.json')

    with open(shortlist_json, encoding='utf-8') as f:
        shortlist_json = json.load(f)

    picture = Category("Picture", ["bafta", "critics", "pga", "sag", "globes"], None, None, 10)
    directing = Category("Directing", ["bafta", "critics", "dga", "globes"])
    leading_actor = Category("Leading Actor", ["bafta", "critics", "sag", "globes"], "Actor")
    leading_actress = Category("Leading Actress", ["bafta", "critics", "sag", "globes"], "Actor")
    supporting_actor = Category("Supporting Actor", ["bafta", "critics", "sag", "globes"], "Actor")
    supporting_actress = Category("Supporting Actress", ["bafta", "critics", "sag", "globes"], "Actor")
    original_screenplay = Category("Original Screenplay", ["bafta", "critics", "wga", "globes"])
    adapted_screenplay = Category("Adapted Screenplay", ["bafta", "critics", "wga", "globes"])
    animated_feature = Category("Animated Feature", ["bafta", "critics", "pga", "annie", "globes"])
    international_feature = Category("International Feature", ["bafta", "critics", "globes"], None, shortlist_json["international_feature"])
    cinematography = Category("Cinematography", ["bafta", "critics", "asc"])
    editing = Category("Editing", ["bafta", "critics", "ace"])
    visual_effects = Category("Visual Effects", ["bafta", "critics", "ves"], None, shortlist_json["visual_effects"])
    score = Category("Original Score", ["bafta", "critics", "globes"], None, shortlist_json["score"])
    song = Category("Original Song", ["critics", "globes"], "Song", shortlist_json["song"])
    production_design = Category("Production Design", ["bafta", "critics", "adg"])
    costume_design = Category("Costume Design", ["bafta", "critics"])
    makeup_hairstyling = Category("Makeup and Hairstyling", ["bafta", "critics"], None, shortlist_json["makeup_hairstyling"])


