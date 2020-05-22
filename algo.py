# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
from nltk.stem.snowball import SnowballStemmer
import math
import json

class tfidf :

    def _step1(self,data,conf,content_normalisation_steps):

        # Detection de l'eventuelle unicite de la langue dans la colonne "language"
        if "language" in data.columns :
            is_unique, language = self._is_language_unique(data)
            if is_unique:
                language = language[0]
            else:
                raise BaseException("This function has not been designed for multilingual analysis")
        else:
            language = conf["language"]

        # Verification que l'id donné est bien une clé primaire.
        if len(data.index) != len(set(data["id_doc"].values)):
            raise BaseException("Column 'id_doc' of the provided DataFrame is not a primary key. Please remove duplicates")
        if type(data["id_doc"].values[0]) != str:
            raise BaseException("Column 'id_doc' of the provided DataFrame must contain exclusively strings")

        # Preparation de la liste des stop words
        stop_words = self._get_stop_words_list(conf, True,data,language)
        # Preparation de la liste des stops stems
        stop_stems_pre  = self._get_word_lists_from_file(language, conf['stopstems_pre_list_file']) if 'stopstems_pre_list_file' in conf.keys() else []
        stop_stems_post = self._get_word_lists_from_file(language, conf['stopstems_post_list_file']) if 'stopstems_post_list_file' in conf.keys() else []

        #######
        # NORMALISATION DU TEXTE
        #######

        # Conversion en minuscules
        all_texts = [st.lower() for st in data["content"].values]
        content_normalisation_steps = content_normalisation_steps + "/to_lower_case"

        # Normalisation du texte + controle du nombre total de characteres
        all_texts, content_normalisation_steps = self._normalise_and_control(all_texts,content_normalisation_steps)

        #######
        # FILTRAGE DES MOTS SELON LEUR FREQUENCE ET LEUR TAILLE
        #######

        # Extraction de tous les mots et de leurs informations de longueur et de position, recuperation des frequences des mots
        all_the_words = self._get_all_the_words(all_texts,data["id_doc"].values)
        freqs         = self._get_word_frequencies(all_the_words)

        return language, stop_words, stop_stems_pre, stop_stems_post, all_the_words, freqs


    def tfidf(self,data,conf,content_normalisation_steps):

        language, stop_words, stop_stems_pre, stop_stems_post, all_the_words, freqs = \
            self._step1(data,conf,content_normalisation_steps)

        if "ngrams_analysis" not in conf.keys() or conf["ngrams_analysis"]:
            ngrams = self._get_ngrams(all_the_words,freqs,conf["bigrams_confidence"],conf)                            
                                                                                         
            all_the_words, freqs = self._integrate_ngrams_to_freqs_and_all_the_words(all_the_words,freqs,ngrams)          
                                                                                                                                                                                                
        # Filtrage
        all_the_words, freqs = self._frequency_and_size_filter(all_the_words, freqs, conf)

        #######
        # PREPARATION DE LA TABLE DE STEMMING
        #######

        replacement = None
        if conf["stemming"]:
            # permet de ne garder que le radical d'un mot
            stemmer = SnowballStemmer(language)

            # remplace un mot X par son radical Y, puis remplace ce radical Y par le mot Z ayant le plus fort taux
            # d'occurrence ayant pour radical Y
            replacement = self._get_replacement_table( freqs,stemmer,stop_stems_pre,stop_stems_post)

        #######
        # TF-IDF
        #######
        idf_detailed = self._get_idf(all_the_words, replacement, stop_words)
        tfidf = self._compute_tfidf(idf_detailed, all_the_words, len(data.index))
        del idf_detailed

        sorted = self._get_sorted(tfidf, all_the_words, replacement, conf["nb_kw_by_text"])

        self.tfidf_dict = tfidf # todo: utile?

        return(sorted)

    def tfidf_incr_add(self,data,conf,content_normalisation_steps,previous_ids, previous_idf= None):

        if set(previous_ids).intersection(set(data["id_doc"])) != set():
            raise BaseException("Some of the provided documents have already been treated: " +
            str(set(previous_ids).intersection(set(data["id_doc"]))))

        language, stop_words, stop_stems_pre, stop_stems_post, all_the_words, freqs = \
        self._step1(data,conf,content_normalisation_steps)

        #######
        # PREPARATION DE LA TABLE DE STEMMING
        #######

        stem_table = None
        if conf["stemming"]:
            # permet de ne garder que le radical d'un mot
            stemmer = SnowballStemmer(language)
            stem_table = self._get_stem_table( freqs,stemmer,stop_stems_pre,stop_stems_post)

        #######
        # TF-IDF
        #######
        idf_detailed = self._get_idf(all_the_words, stem_table, stop_words)
        tfidf = self._compute_tfidf(idf_detailed,all_the_words, len(data.index), previous_idf, len(previous_ids))
        idf_undetailed = self._get_idf_undetailed(idf_detailed) # <m>
        del idf_detailed

        sorted = self._get_sorted(tfidf, all_the_words, stem_table, conf["nb_kw_by_text"])

        self.tfidf_dict = tfidf # todo: utile?

        if previous_idf is None:
            final_idf = idf_undetailed
        else:
            final_idf = self._merge_idfs(idf_undetailed, previous_idf)

        return(sorted, final_idf, previous_ids + list(data["id_doc"]))

    def _get_sorted(self, tfidf, all_the_words, replacement, max_nb_kw):

        # code moche a partir d'ici
        sorted = dict()
        for word in tfidf.keys():
            for doc in tfidf[word].keys():
                sorted = self._add_to_top_n(sorted,tfidf,word,doc,max_nb_kw)

        for doc in all_the_words.keys():
            if doc in sorted.keys(): # par securite
                sorted_doc_keys = sorted[doc].keys()
                for word_num in all_the_words[doc].keys():
                    word = all_the_words[doc][word_num]["word"]
                    try:
                        word = replacement[word]
                    except:
                        pass
                    for word_num_2 in sorted_doc_keys:                         
                        if word == sorted[doc][word_num_2]["word"]:
                            original_word = all_the_words[doc][word_num]["word"]
                            original_len = len(original_word)
                            potition_in_doc = all_the_words[doc][word_num]["pos"]
                            if "occ" in sorted[doc][word_num_2].keys():
                                sorted[doc][word_num_2]["occ"] = sorted[doc][word_num_2]["occ"] \
                                                                   + [dict({"pos": potition_in_doc, "len": original_len})]
                            else:
                                sorted[doc][word_num_2]["occ"] = [dict({"pos": potition_in_doc, "len": original_len})]
        return sorted

    def _get_idf_undetailed(self,idf_detailed):
        idf_undetailed = idf_detailed.copy()
        for word in idf_detailed:
            idf_undetailed[word] = len(idf_detailed[word])
        return idf_undetailed

    def _merge_idfs(self,idf_undetailed, previous_idf):
        sum = idf_undetailed.copy()
        for word in previous_idf:
            if word in sum.keys():
                sum[word] = sum[word] + previous_idf[word]
            else:
                sum[word] = previous_idf[word]
        return sum

    def _add_to_top_n(self,sorted,tfidf,word,doc,max_nb_kw, nb_decimals= 6):
        i = 1
        end = False
        weight = round(tfidf[word][doc], nb_decimals)
        if not doc in sorted.keys():
            sorted[doc] = dict({1: {"word": word, "weight": weight}})
        else:
            while i <= max_nb_kw and not end:
                if i not in sorted[doc].keys():
                    sorted[doc][i] = {"word": word, "weight": weight}
                    end = True
                elif sorted[doc][i]["weight"] < tfidf[word][doc]:
                    aux = sorted[doc][i]
                    sorted[doc][i] = {"word": word, "weight": weight}
                    for j in range(i+1, max(sorted[doc].keys())+1):
                        if j in sorted[doc].keys():
                            aux2 = sorted[doc][j]
                            sorted[doc][j] = aux
                            aux = aux2
                        else:
                            sorted[doc][j] = aux
                    end = True
                i = i+1
        return sorted


    def _get_ngrams(self,all_the_words,freqs,ceil,conf):

        prev,follow = self._get_word_neighbours(all_the_words,freqs,ceil,conf)

        result = dict()
        for key in follow.keys():
            result[key] = dict({follow[key]:None})
        for key in prev.keys():
            word = prev[key]
            if not word in result.keys():
                result[word] = dict()
            result[word][key] = None

        return result

    def _get_stem_table(self,freqs,stemmer,stop_stems_pre,stop_stems_post):

        stem_table = dict()

        for word in freqs:
            replacement_word = self._stem(word,stemmer,stop_stems_pre,stop_stems_post)
            if replacement_word != word:
                stem_table[word] = replacement_word

        return stem_table

    def _get_replacement_table(self,freqs,stemmer,stop_stems_pre,stop_stems_post):
        who_occurred_the_most = dict()
        for word in list(freqs.keys()):
            st_word = self._stem(word,stemmer,stop_stems_pre,stop_stems_post)
            if st_word in list(who_occurred_the_most.keys()):
                if who_occurred_the_most[st_word]["nb_occurr"] < freqs[word]:
                    who_occurred_the_most[st_word] = dict({"best_word": word ,"nb_occurr": freqs[word]})
            else:
                who_occurred_the_most[st_word] = dict({"best_word": word ,"nb_occurr": freqs[word]})

        replacement = dict()
        for word in list(freqs.keys()):
            st_word = self._stem(word,stemmer,stop_stems_pre,stop_stems_post)
            replacement_word = who_occurred_the_most[st_word]["best_word"]
            if replacement_word != word:
                replacement[word] = replacement_word

        return replacement

    def _stem(self,word,stemmer,stop_stems_pre,stop_stems_post):
        if word in stop_stems_pre:
            return word
        st_word = stemmer.stem(word)
        if st_word in stop_stems_post:
            return word
        else:
            return st_word

    def _frequency_and_size_filter(self,all_the_words,freqs,conf):
        # Suppression des mots ayant moins de conf["nchar_min"] caracteres et ayant moins
        for doc_id in list(all_the_words.keys()) :
            for word_id in list(all_the_words[doc_id].keys()) :
                word = all_the_words[doc_id][word_id]["word"]
                if len(all_the_words[doc_id][word_id]["word"]) < conf["nchar_min"] :
                    del all_the_words[doc_id][word_id]
                    if word in freqs.keys():
                        del freqs[word]
                elif freqs[word] < conf["nb_occurr_min"] :
                    del all_the_words[doc_id][word_id]
        return all_the_words,freqs

    def _compute_tfidf(self,idf, all_the_words, total_nb_docs, previous_idf = None, nb_previous_docs= None, norm = True): # todo: cette fonction est salement codée...

        if norm:
            nb_words_by_doc = dict()
            for doc in all_the_words.keys():
                nb_words_by_doc[doc] = len(all_the_words[doc]) # todo : non teste
        else:
            nb_words_by_doc = None

        if previous_idf is None:
            return self._compute_tfidf_without_previous(idf, nb_words_by_doc, total_nb_docs, norm = norm)
        else:
            return self._compute_tfidf_with_previous(idf, nb_words_by_doc, total_nb_docs,previous_idf,nb_previous_docs, norm = norm, max_equals_1= True)

    def _compute_tfidf_without_previous(self,idf, nb_words_by_doc, total_nb_docs, norm, max_equals_1=True):
        tfidf = idf.copy()                                                                                             
        if max_equals_1:
            max_word_weight_by_doc = dict()
        for word in idf.keys():
            nb_docs = len(idf[word])
            for doc in idf[word].keys():
                if norm :
                    tfidf[word][doc] = float(tfidf[word][doc] / nb_words_by_doc[doc]) * math.log(total_nb_docs/nb_docs)
                else:
                    tfidf[word][doc] = float(tfidf[word][doc]) * math.log(total_nb_docs/nb_docs)
                if max_equals_1:
                    if doc in max_word_weight_by_doc.keys():
                        if max_word_weight_by_doc[doc] < tfidf[word][doc] :
                            max_word_weight_by_doc[doc] = tfidf[word][doc]
                    else:
                        max_word_weight_by_doc[doc] = tfidf[word][doc]
        if max_equals_1:
            for word in tfidf.keys():  
                for doc in tfidf[word].keys():
                    tfidf[word][doc] = tfidf[word][doc] / max_word_weight_by_doc[doc]



        return tfidf

    def _compute_tfidf_with_previous(self,idf, nb_words_by_doc, total_nb_docs,previous_idf,nb_previous_docs, norm):
        if type(nb_previous_docs) != int:
            raise BaseException("nb_previous_docs has not been provided.")
        tfidf = idf.copy()                                                                                              
        if max_equals_1:
            max_word_weight_by_doc = dict()
        for word in idf.keys():
            nb_docs = len(idf[word])
            if word in previous_idf.keys():
                nb_docs = nb_docs + previous_idf[word]
            for doc in idf[word].keys():
                if norm :
                    tfidf[word][doc] = float(tfidf[word][doc] / nb_words_by_doc[doc] ) * math.log((total_nb_docs+nb_previous_docs)/nb_docs)
                else:
                    tfidf[word][doc] = float(tfidf[word][doc]) * math.log((total_nb_docs+nb_previous_docs)/nb_docs)
                if max_equals_1:
                    if doc in max_word_weight_by_doc.keys():
                        if max_word_weight_by_doc[doc] < tfidf[word][doc] :
                            max_word_weight_by_doc[doc] = tfidf[word][doc]
                    else:
                        max_word_weight_by_doc[doc] = tfidf[word][doc]
        if max_equals_1:
            for word in tfidf.keys():
                for doc in tfidf[word].keys():
                    tfidf[word][doc] = tfidf[word][doc] / max_word_weight_by_doc[doc]
        return tfidf

    def _get_all_the_words(self,all_texts,text_ids):
        all_the_words    = dict()
        for i in range(len(all_texts)) :
            row_id = text_ids[i]
            row_text = all_texts[i]

            # Extraction de tous les mots du texte en cours
            pointer       = 1
            cpt_words     = 1
            words         = dict()
            for word in re.split("\s",row_text):
                nchar   = len(word)

                # enregistrement du mot, de sa taille, et de sa position dans le texte
                if nchar > 0 :
                    words[cpt_words] = dict({"word": word, "pos": pointer})
                pointer = pointer + 1 + nchar

                # incrementation du compteur
                cpt_words = cpt_words +1

            all_the_words[row_id] = words

        return all_the_words

    def _get_word_frequencies(self, all_the_words):
        word_frequencies = dict()
        for id_doc in all_the_words.keys():
            try:
                nb_words      = max(all_the_words[id_doc])
            except ValueError:
                nb_words = 0
            keys          = list(all_the_words[id_doc].keys())
            for i in range(nb_words +1):
                if i in keys:
                    word = all_the_words[id_doc][i]["word"]
                    if len(word) > 0 :                                                                                 
                        # Si le mot a deja ete traite (quelque soit la ligne)
                        if word in word_frequencies.keys():
                            word_frequencies[word] = word_frequencies[word] +1
                        else:
                            word_frequencies[word] = 1
        return word_frequencies

    def _get_idf(self, all_the_words,replacement=None, stop_words = []):
        if replacement != None:
            replacement_keys = replacement.keys()
        else:
            replacement_keys = []

        idf = dict()
        for id_doc in all_the_words.keys():
            try:
                nb_words      = max(all_the_words[id_doc])
            except ValueError:
                nb_words = 0
            keys          = list(all_the_words[id_doc].keys())
            for i in range(nb_words +1):
                if i in keys:
                    word = all_the_words[id_doc][i]["word"]
                    if word in replacement_keys:
                        word = replacement[word]
                    if word in idf.keys():
                        if id_doc in idf[word].keys():
                            idf[word][id_doc] = idf[word][id_doc]+1
                        else:
                            idf[word][id_doc] = 1
                    else:
                        idf[word] = dict()
                        idf[word][id_doc] = 1

        for sw in stop_words:
            try:
                del idf[sw]
            except:
                pass

        return idf

    def _get_idf2(self, all_the_words,replacement=None, stop_words = []):
        if replacement != None:
            replacement_keys = replacement.keys()
        else:
            replacement_keys = []

        idf = dict()
        for id_doc in all_the_words.keys():
            nb_words      = max(all_the_words[id_doc])
            keys          = list(all_the_words[id_doc].keys())
            for i in range(nb_words +1):
                if i in keys:
                    word = all_the_words[id_doc][i]["word"]
                    if word in replacement_keys:
                        word = replacement[word]
                    if word in idf.keys():
                        idf[word] = idf[word] + 1
                    else:
                        idf[word] = 1

        for sw in stop_words:
            try:
                del idf[sw]
            except:
                pass

        return idf


    def _integrate_ngrams_to_freqs_and_all_the_words(self,all_the_words,freqs,ngrams):
        if ngrams != dict():
            for id_doc in all_the_words.keys():
                try:
                    nb_words      = max(all_the_words[id_doc])
                except ValueError:
                    nb_words = 0
                route = []
                route_indices = []
                sauv_i = 0
                i = 0
                end_of_document = False
                while i <= nb_words or end_of_document:                                                                 # il faut prendre les clés dans l'ordre. Dans le cas d'un all_the_words.keys(), les cles ne seraient pas forcement dans l'ordre
                    if i in all_the_words[id_doc].keys() or end_of_document:
                        if not end_of_document:
                            word = all_the_words[id_doc][i]["word"]
                        else:
                            word = ""

                        word_dict = self._get_filtered_all_the_words_from_route(ngrams,route)
                        if route == []:
                            if word in word_dict.keys():
                                route = [word]
                                route_indices = [i]
                                sauv_i = i
                        else :
                            if word_dict == None:
                                self._merge_words(route,route_indices,all_the_words,id_doc)
                                self._modify_freqs(route,freqs)
                                i = sauv_i
                                route = []
                                route_indices = []
                            else:
                                if not end_of_document:
                                    if word in word_dict.keys():
                                        route = route + [word]
                                        route_indices = route_indices + [i]
                                    else:
                                        i = sauv_i
                                        route = []
                                        route_indices = []
                    i = i+1

                    if i > nb_words:
                        if i == nb_words+1 and route != []:  # si l'on n'a pas fini de traiter route alors que l'on arrive à la fin du document
                            end_of_document = True
                        else:
                            end_of_document = False

        return all_the_words,freqs

    def _merge_words(self,route,route_indices,all_the_words,doc_id):
        first_word_indice = route_indices[0]
        pos_first_word_indice = all_the_words[doc_id][first_word_indice]["pos"]
        for i in route_indices:
            del all_the_words[doc_id][i]
        new_word = ""
        for word in route:
            new_word = new_word + word + " "
        new_word = new_word[0:-1]
        all_the_words[doc_id][first_word_indice] = dict({"word":new_word,"pos":pos_first_word_indice})


    def _modify_freqs(self,route,freqs):
        new_word = ""
        for word in route:
            freqs[word] = freqs[word] - 1
            if freqs[word] == 0:
                del freqs[word]
            new_word = new_word + word + " "
        new_word = new_word[0:-1]

        if new_word in freqs.keys():
            freqs[new_word] = freqs[new_word] + 1
        else:
            freqs[new_word] = 1


    def _get_filtered_all_the_words_from_route(self,word_dict,route):
        result = word_dict
        for key in route:
            result = result[key]
        return result



    def _get_word_neighbours(self,all_the_words,freqs,ceil,conf):
        preceeding_word  = dict()
        following_word   = dict()
        for id_doc in all_the_words.keys():
            #print(id_doc)
            previous_word = None
            try:
                nb_words      = max(all_the_words[id_doc])
            except ValueError:
                nb_words = 0
            keys          = list(all_the_words[id_doc].keys())
            for i in range(nb_words +1):
                # si le mot numero i n'a pas été filtré auparavent (en dehors de cette fonction)
                if i in keys:
                    word = all_the_words[id_doc][i]["word"]
                    is_preceeding_word_modified = False
                    is_following_word_modified = False
                    if len(word) > 0 :                                                                                  # TODO: normalement innutile si all_the_words ne contient pas de ""
                        # Si le mot a deja ete traite
                        if word in preceeding_word.keys():
                            # si le mot n'a pas encore été filtré
                            if preceeding_word[word] != None:
                                # Si le mot precedent a deja ete repere comme predecesseur
                                if previous_word in preceeding_word[word].keys():
                                    # on incremente de 1 le nombre d'occurrences
                                    preceeding_word[word][previous_word] = preceeding_word[word][previous_word] +1
                                    is_preceeding_word_modified = True
                                else:
                                    # on initialise a 1
                                    preceeding_word[word][previous_word] = 1
                                    is_preceeding_word_modified = True
                        else:
                            if previous_word != None:
                                # on initialise
                                preceeding_word[word]  = dict({previous_word:1})
                                is_preceeding_word_modified = True

                        # copié collé des lignes précédentes avec adaptation
                        if previous_word in following_word.keys():
                            if following_word[previous_word]!= None:
                                if word in following_word[previous_word].keys():
                                    following_word[previous_word][word] = following_word[previous_word][word]+1
                                    is_following_word_modified = True
                                else:
                                    following_word[previous_word][word] = 1
                                    is_following_word_modified = True
                        elif previous_word != None:
                            if word != None:
                                following_word[previous_word] = dict({word:1})
                                is_following_word_modified = True                                                                      # TODO:voir si on peut en supprimer quelques uns

                        # si le mot A est precede dans : 20% des cas du mot B, 17% des cas du mot C, 63% des cas d'autres
                        # mots pas encore detectés à l'itération courrante, et qu'il est souhaité que le bigramme M-A
                        # apparaisse dans 90% des cas où A apparait, alors ni B, ni C, ni aucun autre mot ne peut
                        # satifaire cette condition. (20%+63%=83%: B ne satisfait pas la condition. 17%+63%=80%: C non
                        # plus. 63% d'autres mots seront toujours inferieurs à 90%) :
                        if is_preceeding_word_modified:
                            self._filter_predecessors(word,preceeding_word,freqs,ceil)
                        if is_following_word_modified:
                            self._filter_predecessors(previous_word,following_word,freqs,ceil)

                        previous_word    = word

        preceeding_word = self._filter_preceeding_word(preceeding_word,ceil,freqs,conf)
        following_word  = self._filter_preceeding_word(following_word,ceil,freqs,conf)

        return preceeding_word, following_word



    def _filter_predecessors(self,word,word_dict,freqs,ceil):
        if word != None:
            frequencies = []
            for prec_word in word_dict[word].keys():
                frequencies = frequencies + [word_dict[word][prec_word]]
            nb_of_unknown_words = freqs[word]- np.array(frequencies).sum()
            if (np.array(frequencies).max() + nb_of_unknown_words)/freqs[word] < ceil:
                word_dict[word] = None

    def _filter_preceeding_word(self,word_list,ceil,freqs,conf):
        result = dict()
        for key in list(word_list.keys()):
            if word_list[key] != None:
                for word in word_list[key].keys():
                    occ = word_list[key][word]
                    if occ >= (ceil*freqs[key]) and occ >= conf["nb_occurr_min"]:
                        result[key] = word
        return result

    def _normalise_and_control(self,all_texts,content_normalisation_steps):
        total_nchar = self._get_number_of_chars(all_texts)
        # content_normalisation_state est ici a titre indicatif. Sert a connaitre la succession de normalisations
        # appliquees au texte
        content_normalisation_steps = content_normalisation_steps + "/textnorm.norm_alphanum(content,additional = '.\-_')"
        #normed_content = [textnorm().norm_alphanum(text) for text in data["content"].values]           
        normed_content = [textnorm().norm_alphanum(text,additional=".\-_") for text in all_texts]        
        normed_content = [re.sub("\.([^\w]|\Z)"," \\1",st) for st in normed_content]              

        if self._get_number_of_chars(normed_content) != total_nchar:
            raise BaseException("The normalisation function shouldn't modify the total number of characters")
        return normed_content, content_normalisation_steps


    def _get_number_of_chars(self,text_list):
        return np.array([len(text) for text in text_list ]).sum()

    def _get_stop_words_list(self,conf,unique_language,data,language):
        SW = conf['spec_stopw']
        if unique_language and conf['use_traditionals_stopw']:
            SW = SW + (self._get_word_lists_from_file(language,conf['language_stopw_list_file']) if 'language_stopw_list_file' in conf.keys() else [])
        #else : # non utilise
        #    SW = [SW + self._get_stop_words( lang, conf['language_stopw_list_file']) for lang in data["language"] ]         # TODO : non teste
        return SW


    def _is_language_unique(self,data):
        languages = data.language.unique()
        if len(languages) == 1:
            return True  , languages
        else:
            return False , languages

    def _get_word_lists_from_file(self,language,file):
        try :
            f = open(file,encoding="utf-8").read()
        except BaseException as e:
            raise BaseException("Error in reading the word list : "+file+"\n "+str(e))
        try :
            words = eval(f)
        except :
            raise BaseException("The file "+ file +""" seems not to be a valid Python file containing (exclusively) a
            dict (not stored in a variable)""")
        try :
            swl = words[language]
        except :
            raise BaseException("Language "+ language +" is not a valid language for the word list of "+ file)
        return swl

    #todo: certains edges sont en doublons (en sens opposé)
    #revoir

class textnorm:
    def norm_alphanum(self, string, additional= ""):
        return re.sub("[^\w"+ additional +"]", " ", string, flags= re.UNICODE)
    
    
if __name__ == "__main__":
    conf = {
        'language': 'french',
        'spec_stopw': [], #stop words spécifiques
        'use_traditionals_stopw' : False, # sinon il faut remplir l aligne du dessous
        'language_stopw_list_file': None, # cf ligne d'apres
        #'stopstems_pre_list_file': , #  conf file contenant uniquement un dict: {'french': ['est','a'], 'english':[...]}
        #'stopstems_post_list_file': , # Si pas de fichier, ne pas inclure la clé (ne pas mettre None)
        "ngrams_analysis":False,
        "nchar_min":3,
        "nb_occurr_min":0, # Peut causer une ZeroDivisionError si trop élevé 
        "stemming": True, 
        "nb_kw_by_text":2
    }
    
    texts = [
        "tableau carton genou choux choux bol",
        "tableau chaise carton genou choux choux bol plante plante",
        "tableau chaise carton choux bol tapis plante",
        "tableau chaise carton genou choux choux bol plante"
    ]

    data = pd.DataFrame({'id_doc': [str(a) for a in range(len(texts))], 'content':texts})

    t = tfidf()

    data.index = data.id_doc

    for id_doc, words in t.tfidf(data,conf,"").items():
        data.loc[id_doc,"kw"] = ",".join([word['word'] for i,word in words.items()])

    print(data)


