#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from num2words import num2words
import itertools
from itertools import combinations
from itertools import permutations
from sympy import Matrix
from diophantine import solve
import pandas as pd
from alphabet_detector import AlphabetDetector

class Vocabulary:
    def __init__(self, nb, nal):
        if type(nb) is list or type(nb) is np.array:
            try:
                nb=nb.item()
            except:
                pass
               # print(type(nb))
                #try:
                    #print(type(nb.item()))
                #except:
                    #pass
                #raise TypeError("Number has to be an integer")
        if not type(nal) is str:
            raise TypeError("Numeral has to be a string")
        self.number = nb
        self.word = nal
        self.root = nal
        self.inputrange = []
        self.mapping = [nb]
    def printVoc(self):
        print(str(self.number)+' '+self.numeral)
    def all_outputs(self):
        return [self]
    def dimension(self):
        return 0
    def actual_dimension(self):
        return 0
    def sample(self):
        return self
class Highlight:
    def __init__(self, voc, start):
        self.number=voc.number
        self.numeral=voc.word
        self.start=start
        self.root=voc.word
        self.mapping=[voc.number]
    def end(self):
        return self.start+len(self.numeral)
    def hlrange(self):
        return range(self.start+1,self.end())
    def voc(self):
        return Vocabulary(self.number,self.numeral)
    
class SCFunction:
    '''
    Root is the exponent of the function where _ mark input slots
    Inputrange is a list of lists. The nth lists all SCFunctions that may enter the nth input slot
    Mapping is a list of coefficients. The nth coefficient is the factor by which the nth input would have to be multiplied
        Exception: the last coefficient is the constant coefficient.
    '''
    def __init__(self,root,i,mapping):
        if not type(root) is str:
            raise TypeError("Root has to be a string")
        if not type(i) is list:
            raise TypeError("Inputrange has to be a list")
        dimension=root.count('_')
        if not len(i) == dimension:
            print(i)
            raise TypeError(str(dimension)+"-dimensional function needs "+str(dimension)+" component domains.")
        for component in i:
            if not type(component) is list:
                print(type(component))
                raise TypeError("All component domains have to be lists")
            for entry in component:
                if not type(entry) is Vocabulary and not type(entry) is SCFunction:
                    print(type(entry))
                    raise TypeError("All entries of all input components have to be Vocabulary or SCFunction")
        if not type(mapping) is list or dimension+1!=len(mapping):
            print(type(mapping))
            try:
                print(len(mapping))
            except:
                pass
            raise TypeError("Mapping has to be a list of "+str(dimension)+"+1 coefficients")
        #for coeff in mapping:
            #if not type(coeff) is int and not type(coeff) is float:
                #print(type(coeff))
                #raise TypeError("All coefficients of the mapping have to be integers or floats")
        self.root=root
        self.inputrange=i
        self.mapping=mapping
    def dimension(self):
        '''
        Dimension = Number of input slots
        '''
        return self.root.count('_')
    def number_inputs(self):
        ni = []
        for comp in self.inputrange:
            ni += [[entr.sample().mapping[-1] for entr in comp]]
        return ni
    def input_numberbase(self):
        build_base=[]
        #print(len(self.inputrange))
        base_complete = False
        for root_inputx in cartesian_product(self.inputrange):
            for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                #print(build_base)
                #print([component.number for component in final_inputx]+[1])
                #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                    #print(str(self.insert(inputx).number)+' is linear independent')
                    build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                    #print([self.insert(inputx).number])
                if len(build_base)==self.dimension()+1:
                    #print('base complete')
                    base_complete = True
                    break
            if base_complete:
                break
        return build_base
        
    def actual_dimension(self):
        '''
        Dimension of input range with respect to affine linearity
        '''
        numbers = [[self.inputrange[i][j].all_outputs()[0].mapping[-1] for j in range(len(self.inputrange[i]))] for i in range(self.dimension())]
        return np.linalg.matrix_rank(np.array(cartesian_product(numbers+[[1]]), dtype=np.float64))
    def insert(self,inputx):
        '''
        Requires a dimension-long list of input SCFunctions
        Return a new SCFunction where all input SCFunctions are inserted in their respective slot
        Updates inputrange and mapping with respect to new inputslots originating from the input SCFunctions
        '''
        
        # catch errors
        if not type(inputx) is list:
            print(type(inputx))
            raise TypeError('Input has to be a list')
        if len(self.inputrange) != len(inputx) or len(self.inputrange) == 0:
            raise TypeError("Input does not match dimension or "+self.root+" has no inputslots.")
        for component in inputx:
            if not type(component) is Vocabulary and not type(component) is SCFunction:
                print(type(component))
                raise TypeError('All components of the input have to be a Vocabulary or an SCFunction')
                
        # trouble shoot if input is not in the inputrange
        for entry in range(len(inputx)):
            if inputx[entry].root not in [inputfunction.root for inputfunction in self.inputrange[entry]]:
                #print("Input "+str([inp.root for inp in inputx])+" is not in the input range of "+str(self.root))
                break
                
        # initialize root, inputrange and mapping of composed SCF
        rootparts = self.root.split('_')
        output_root = ''
        output_i = []
        output_mapping = []
        constant_coefficient = self.mapping[-1]
        
        # extend root, inputrange and mapping
        for inp in range(len(inputx)):
            output_root += rootparts[inp] + inputx[inp].root
            output_i += inputx[inp].inputrange
            output_mapping += [self.mapping[inp] * coeff for coeff in inputx[inp].mapping[:-1]]
            constant_coefficient += self.mapping[inp] * inputx[inp].mapping[-1]
            
        # finish root and mapping and return composed SCF
        output_root += rootparts[-1]
        output_mapping += [constant_coefficient]
        return SCFunction(output_root,output_i,output_mapping)
    def sample(self):
        if self.dimension() == 0:
            return self
        else:
            return self.insert([comp[0].sample() for comp in self.inputrange])
    
    def all_outputs(self):
        '''
        return all final SCFunctions (vocabulary) without unsatisfied '_'s left, that are derivable from 
        '''
        #print('alloutputs of '+self.root)
        if self.dimension() == 0:
            return [self]
        else:
            all_output = []
            #print(self.inputrange)
            for inputvector in cartesian_product(self.inputrange):
                new_outputs = self.insert(inputvector).all_outputs()
                all_output += new_outputs
            return all_output
    def all_outputs_as_voc(self):
        ao = self.all_outputs()
        aov = []
        for scf in ao:
            aov += [Vocabulary(scf.mapping[-1],scf.root)]
        return aov
    
    def merge(self,mergee):
        if not type(mergee) is SCFunction:
            print('not scf')
            raise TypeError("Can only merge with other SCFunction") 
        if not self.root==mergee.root:
            print('different roots')
            raise BaseException('Cannot merge with SCFunction with different exponent')
        if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
            print('mergee is not singleton')
            raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
        if self.dimension() == 0:
            print('merger is not generalizable')
            raise BaseException('SCFunction of dimension 0 cannot merge')
        #if mergee.insert(mergee.inputrange[0]).number!=self.insert(mergee.inputrange[0]).number:
            #print('EXPERIMENTAL ERROR! The constructed SCFunction is not affine linear')
        #if [component.number for component in mergee.inputrange[0]] in SPAN(self.inputrange): # insert(Mergee)=mergee.number
        new_inputrange = []
        for comp in range(self.dimension()):
            if mergee.inputrange[comp][0].root in [ent.root for ent in self.inputrange[comp]]:
                new_inputrange += [self.inputrange[comp]]
            else:
                new_inputrange += [self.inputrange[comp] + mergee.inputrange[comp]]
        insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
        if insert.mapping[-1] == mergee.mapping[-1]:
            #print('Current mapping predicts value of '+insert.root+' correctly')
            return SCFunction(self.root,new_inputrange,self.mapping)
        else:
            # DANN PRÜFE ERST OB MERGEE NICHT IM SPANN DER INPUTRANGE LIEGT. WENN DOCH, DANN BRAUCHT ES EINE SEPARATE FUNKTION
            # MACH PROPOSAL UND DANN PRÜFE OB ES EINE HÖHERE DIMENSION HAT ALS SELF. WENN NICHT KANN ES NICHT MERGEN
            build_base = self.input_numberbase()
            build_image = [np.dot(self.mapping,basevec) for basevec in build_base]
            #print([[component[0].number for component in mergee.inputrange]+[1]] + build_base)
            new_dim = np.linalg.matrix_rank(np.array([[component[0].number for component in mergee.inputrange]+[1]] + build_base, dtype=np.float64))
            #print('newdim determined')
            if new_dim > self.actual_dimension():
                #print('expand base and image')
                build_base = [[component[0].number for component in mergee.inputrange]+[1]] + build_base
                build_image = [mergee.mapping[-1]] + build_image
            else:
                print('No merge')
                print(build_base)
                print([component[0].number for component in mergee.inputrange]+[1])
                raise BaseException('Mergee must have a different mapping')
            '''
            build_base=[[component[0].number for component in mergee.inputrange]+[1]]
            build_image=[mergee.mapping[-1]]
            #print(len(self.inputrange))
            base_complete = False
            for root_inputx in cartesian_product(self.inputrange):
                for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                    #print(build_base)
                    #print([component.number for component in final_inputx]+[1])
                    #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                    if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                        #print(str(self.insert(inputx).number)+' is linear independent')
                        build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                        #print([self.insert(inputx).number])
                        build_image=build_image+[self.insert(final_inputx).mapping[-1]]
                    if len(build_base)==self.dimension()+1:
                        #print('base complete')
                        base_complete = True
                        break
                if base_complete:
                    break
            '''
            #print('calculation')
            #print(build_base)
            #print(build_image)
            try:
                coefficients=solve(build_base,build_image)[0]
            except:
                coefficients=np.dot(np.linalg.pinv(np.array(build_base, dtype=np.float64),rcond=1e-15),build_image)
            coefficients=[round(coeff) for coeff in coefficients]
            #print(coefficients)
            #print(len(mergee.inputrange+self.inputrange))
            return SCFunction(self.root,new_inputrange,list(coefficients))
    def present(self):
        if self.dimension() == 0:
            print(self.root+" is "+str(self.mapping[-1]))
        else:
            domainstrs = []
            for comp in self.inputrange:
                component = '{'
                for entry in comp:
                    if isinstance(entry,Vocabulary) or entry.dimension() == 0:
                        component += str(entry.mapping[-1])+','
                    else:
                        component += str(entry.root)+','
                component = component[:-1]+'}'
                domainstrs += [component]
            domainstr = 'x'.join(domainstrs)
            if self.dimension() == 1:
                inpstr = 'x'
                outpstr = str(self.mapping[0]) + '*x+' + str(self.mapping[1])
            else:
                inpstr = '('
                outpstr = ''
                for comp in range(self.dimension()):
                    inpstr += 'x'+str(comp)+','
                    outpstr += str(self.mapping[comp]) + '*x' + str(comp) + '+'
                inpstr = inpstr[:-1] + ')'
                if self.mapping[-1] != 0:
                    outpstr += str(self.mapping[-1])
                else:
                    outpstr = outpstr[:-1]
            retstr = "Function " + self.root + " maps " + domainstr + " by " + inpstr + ' -> ' + outpstr
            print(retstr)
    def reinforce(self,lexicon,supervisor):
        copy = self
        candidates_for_abstraction = []
        upper_limit = sum([max([0,coeff]) for coeff in self.mapping]) - 1
        print(upper_limit)
        for entry in lexicon:
            if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs()]):
                print(entry.root)
                print([fe.mapping[-1] for fe in entry.all_outputs()])
                candidates_for_abstraction += [entry]
        invariant_slots = []
        for comp in range(copy.dimension()):
            if len(copy.inputrange[comp]) == 1:
                invariant_slots += [comp]
        for combination in cartesian_product((copy.dimension() - len(invariant_slots)) * [candidates_for_abstraction]):
            cand = []
            for comp in range(copy.dimension()):
                if comp in invariant_slots:
                    cand += [copy.inputrange[comp][0]]
                else:
                    cand += [combination[0]]
                    combination = combination[1:]
            word_cand = copy.insert([candc.sample() for candc in cand])
            #print(word_cand.root)
            proposed_inputrange = []
            entry_is_new = False
            for comp in range(copy.dimension()):
                if cand[comp].root in [entr.root for entr in copy.inputrange[comp]]:
                    proposed_inputrange += [copy.inputrange[comp]]
                else:
                    proposed_inputrange += [copy.inputrange[comp]+[cand[comp]]]
                    entry_is_new = True
            if entry_is_new:
                proposal = SCFunction(copy.root,proposed_inputrange,copy.mapping)
                proposal_is_new = True
                for entr in lexicon:
                    for word in entr.all_outputs():
                        if word_cand.root == word.root or word_cand.mapping[-1] == word.mapping[-1]:
                            #print('proposal ' + word_cand.root + ' ' + str(word_cand.mapping[-1]) + ' already covered by ' + word.root + ' ' + str(word.mapping[-1]))
                            proposal_is_new = False
                            break
                    if not proposal_is_new:
                        break
                if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                    print('Can I also say '+ word_cand.root + '?')
                    for v in supervisor:
                        if word_cand.root == v.word:
                            print('Supervisor: Yes')
                            cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,lexicon,False,False)
                            if cand_parse.root == self.root:                                
                                if v.number != word_cand.mapping[-1]:
                                    print('LEARNING ERROR: ' + v.word + ' is ' + str(v.number) + ' but learner assumes ' + str(word_cand.mapping[-1]))
                                copy = proposal
                                copy.present()
                            else:
                                pass
                                print('OK, but I think this is not related')
                            break
                    else:
                        pass
                        print('Supervisor: No')
        return copy

def cartesian_product(listlist):
    if len(listlist)==0:
        print('ERROR: empty input in product')
        return False
    for liste in listlist:
        if not isinstance(liste,list):
            liste=[liste]
    cp=[[x] for x in listlist[0]]
    for liste in listlist[1:]:
        cp=[a+[b] for a in cp for b in liste]
    return cp

def delatinized(string):
    #print(string)
    ad = AlphabetDetector()
    if not ad.is_latin(string):
        if not ad.is_cyrillic(string):
            if ad.is_cyrillic(string[0]) and not ad.is_cyrillic(string[-1]):
                #print('first part is cyrillic')
                for point in range(len(string)):
                    if not ad.is_cyrillic(string[:point]):
                        return string[:point-2]
            elif not ad.is_cyrillic(string[0]) and ad.is_cyrillic(string[-1]):
                #print('last part is cyrillic')
                for point in reversed(range(len(string))):
                    if not ad.is_cyrillic(string[point:]):
                        return string[point+2:]
            elif ad.is_latin(string[0]):
                #print('first part is latin')
                for point in range(len(string)+1):
                    if not ad.is_latin(string[:point]):
                        return string[point-1:] 
            elif ad.is_latin(string[-1]):
                #print('last part is latin')
                for point in reversed(range(len(string)+1)):
                    if not ad.is_latin(string[point:]):
                        return string[:point+1]
            else:
                return string
        else:
            return string
    else:
        return string

def create_lexicon(language):
    LEX=[]
    try:
        num2words(1, lang=language)
        for integer in list(range(1,1001))+[1002,1006,1100,1200,1206,7000,7002,7006,7100,7200,7206,10000,17000,17206,20000,27000,27006,27200,27206]:
            try:
                numeral=num2words(integer, lang=language)
                voc=Vocabulary(integer,numeral)
                LEX=LEX+[voc]
            except:
                pass
        return LEX
    except:
        try:
            lanu=pd.read_csv(r'C:\Users\ikm\OneDrive\Desktop\NumeralParsingPerformance\Languages&NumbersData\Numeral.csv', encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            for i in range(len(df)):
                numeral=df.iloc[i,2]
                if numeral[0]==' ':
                    numeral=numeral[1:]
                if numeral[-1]==' ':
                    numeral=numeral[:-1]
                if language in ['Latin','Persian','Arabic']:
                    words=numeral.split(' ')
                    numeral=' '.join(iter(words[:-1]))
                if language in ['Chuvash','Adyghe']:
                    words=numeral.split(' ')
                    numeral=' '.join(iter(words[:len(words)//2]))
                if biscriptual and not language in ['Chuvash','Adyghe','Latin','Persian','Arabic']:
                    numeral=delatinized(numeral)
                #print(numeral)
                numeral=numeral.replace('%',',')
                #print(numeral)
                voc=Vocabulary(i+1,numeral)
                LEX=LEX+[voc]
            return LEX
        except:
            raise NotImplementedError("Language "+language+" is not supported or spelled differently")


def advanced_parse(number, word, lexicon, print_doc, print_result):
    if print_doc: print('parse '+word+' '+str(number))
    lexicon1 = []
    if len(lexicon) != 0 and isinstance(lexicon[0],Vocabulary):
        lexicon1 = lexicon
    else:
        for entry in lexicon:
            if isinstance(entry,Vocabulary):
                lexicon += [entry]
            else: 
                lexicon1 += entry.all_outputs_as_voc()
    lexicon1 = lexicon1+[Vocabulary(number,word)]    
    checkpoint = 0
    highlights=[]
    mult_found=False
    for end in range(0, len(word)+1):
        startrange=set(range(checkpoint, end))
        for highlight in highlights:
            startrange=startrange-set(highlight.hlrange())
            #print('remove '+str(range(highlight[3]+1,len(highlight[0]))))
        startrange=sorted(list(startrange))
        #print('startrange='+str(startrange))
        for start in startrange:
            subnum_found_at_this_end=False
            substr=word[start:end]
            #if print_doc: print('substring = '+str(substr))
            for entry in lexicon1:
                #if printb: print('lexnum = '+str(numeral[0]))
                if substr == entry.word:
                    subnum_found_at_this_end=True
                    subentry_found = False
                    if 2*entry.number < number or mult_found:
                        for highlight in reversed(highlights):
                            if highlight.start >= start:
                                if print_doc: print("remove "+highlight.numeral)
                                highlights.remove(highlight)
                        highlights=highlights+[Highlight(Vocabulary(entry.number,entry.word),start)]
                        if print_doc: print([highlight.numeral for highlight in highlights])
                    else: 
                        if print_doc: print(substr+' violates HC')
                        mult_found=True
                        checkpoint=end
                        potential_highlight = None
                        earliest_laterstart = start+1
                        for highlight in highlights:
                            if highlight.number**2 < entry.number:
                                earliest_laterstart = min(end,highlight.end()) #so factors remain untouched
                        for laterstart in range(earliest_laterstart,end):
                            if print_doc: print('subnum = '+word[laterstart:end])
                            for subentry in lexicon1:
                                if word[laterstart:end] == subentry.word:
                                    if subentry.number**2 <= entry.number:
                                        if print_doc: print(word[laterstart:end]+" is FAC or SUM. If it would contain mult, its square would be larger than "+entry.word+'.')
                                        subentry_found=True
                                        for highlight in reversed(highlights):
                                            if highlight.end() > laterstart:
                                                if print_doc: print("remove "+highlight[0])
                                                highlights.remove(highlight)
                                        highlights=highlights+[Highlight(Vocabulary(subentry.number,subentry.word),laterstart)]
                                        checkpoint = laterstart
                                        potential_highlight = None
                                        if print_doc: print([highlight.numeral for highlight in highlights])
                                    else:
                                        if entry.number % subentry.number != 0 and 2*subentry.number < number:
                                            if print_doc: print(word[laterstart:end]+" probably contains SUM. As "+subentry.word+' is no divisor of '+entry.word+', '+entry.word+' has to contain SUM. '+subentry.word+' cannot contain FAC*MULT, as it is smaller than half of '+entry.word+': And it cannot be FAC, as its square is larger than '+entry.word+'. So it is composed of SUM and MULT. If it turns out to be irreducible with the present properties, we assume it is SUM')
                                            potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                            potential_checkpoint = laterstart
                                        else:
                                            potential_highlight = None
                            if subentry_found:
                                break  
                        if potential_highlight != None:
                            for highlight in reversed(highlights):
                                if highlight.end() > potential_checkpoint:
                                    if print_doc: print("remove "+highlight.word)
                                    highlights.remove(highlight)
                            highlights = highlights+[potential_highlight]
                            checkpoint = potential_checkpoint
                            if print_doc: print([highlight.word for highlight in highlights])
                    break                    
            if subnum_found_at_this_end:
                break
    if len(highlights) == 2:
        if highlights[0].number + highlights[1].number == number or highlights[0].number * highlights[1].number == number:
            suspected_mult = max(highlights, key=lambda highlight: highlight.number)
            if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
            highlights.remove(suspected_mult)
    elif len(highlights) == 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            if other_numbers[0] * suspected_mult.number + other_numbers[1] == number or other_numbers[1] * suspected_mult.number + other_numbers[0] == number:
                if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
                highlights.remove(suspected_mult)
    elif len(highlights) > 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            for suspected_factor in other_numbers:
                suspected_summand = sum(other_numbers)-suspected_factor
                if suspected_factor * suspected_mult.number + suspected_summand == number:
                    if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
                    highlights.remove(suspected_mult)
                    break
        
    #print(str(highlights)+wort+' '+str(zahl))
    root=word
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr = str(number)+'='+root+'('+','.join([str(highlight.number) for highlight in highlights])+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])


def learn_language(language):
    print('Learning '+language)
    supervisor = create_lexicon(language)
    learnerlex = []
    samples = 0
    for voc in supervisor:
        if not any([voc.word in [outp.root for outp in entr.all_outputs()] for entr in learnerlex]):
            samples += 1
            #1 parse new word
            print('What means '+str(voc.number)+'?')
            print('Supervisor: '+voc.word)#+' means '+str(voc.number))
            parse = advanced_parse(voc.number, voc.word, learnerlex, False, True)
            # Understood?
            understood = False
            for entry in learnerlex:
                if entry.root == parse.root:
                    #2a Yes, assuming to have understood. Trying to update an entry
                    try:
                        learned = entry.merge(parse)
                        learnerlex.remove(entry)
                        learned.present()
                        #print(str(entry.actual_dimension()) + ' < ' + str(learned.actual_dimension()) + ' ?')
                        if entry.actual_dimension() < learned.actual_dimension():
                            print('Attempting reinforcement')
                            learned = learned.reinforce(learnerlex,supervisor)
                        understood = True
                        break
                    except:
                        print('Not related to '+entry.root)
                        pass
            if not understood:
                #2b No, just remembering
                learned = parse
            learned.present()
            learnerlex += [learned]
    #reorganize all inputranges so that they only contain scfunctions, no vocabulary
    new_learnerlex = []
    for lexentr in learnerlex:
        lexentr.present()
        new_inputrange = []
        for comp in lexentr.inputrange:
            new_comp = []
            for entr in comp:
                if isinstance(entr,SCFunction):
                    if not entr in new_comp:
                        new_comp += [entr]
                elif isinstance(entr,Vocabulary):
                    parse = advanced_parse(entr.number,entr.word,learnerlex,False,False)
                    for scf in learnerlex:
                        if isinstance(scf,SCFunction) and parse.root == scf.root and entr.root in [op.root for op in scf.all_outputs()]:
                            if not scf in new_comp:
                                new_comp += [scf]
            new_inputrange += [new_comp]
        new_learnerlex += [SCFunction(lexentr.root,new_inputrange,lexentr.mapping)]
    learnerlex = new_learnerlex
            
    print('Learned '+str(len(supervisor))+' words in '+language+' and structured them in '+str(len(learnerlex))+' functions.')
    print('It took '+str(samples)+' samples to learn those.')
    print('Those are:')
    for entry in learnerlex:
        entry.present()
    print('')


# In[2]:


import time

def scfbigger(x,y):
    '''
    return 1/0/-1 if x is bigger/equal/smaller than y
    '''
    xnumbers = [op.mapping[-1] for op in x.all_outputs()]
    ynumbers = [op.mapping[-1] for op in y.all_outputs()]
    if min(xnumbers) > max(ynumbers):
        return 1
    elif min(ynumbers) > max(xnumbers):
        return -1
    else:
        return 0
  

def scflist2cfg(scflist):
    #print('Create CFG')
    # Categorize scfunctions
    # by creating a list of lists where each list is a disjoint set of scfs that form a category
    for scf in scflist:
        if not isinstance(scf,SCFunction):
            raise BaseException('Grammar contains objects other than SCFunctions')
    categorization = []
    copy = [scf for scf in scflist]
    start_time = time.time()
    while len(copy) > 0:
        next_highest_categ = []
        biggest_input = SCFunction('',[],[-10000])
        for scf in copy:
            for comp in scf.inputrange:
                for entr in comp:
                    if scfbigger(entr,biggest_input) == 1:
                        biggest_input = entr
        for scfu in reversed(copy):
            if scfbigger(scfu,biggest_input) == 1:
                next_highest_categ += [scfu]
                copy.remove(scfu)
        categorization = [next_highest_categ] + categorization
        if time.time() - start_time > 60:
            print('Could not categorize')
            print('Remaining functions are: ',end='')
            print([f.root for f in copy])
            print('Biggest remaining input is:',end='')
            biggest_input.present()
            raise BaseException('Grammar cannot becategorized due to hierarchy loops')
            
    #print('SCFunctions categorized')
    # determine list of NTs
    inputsets = []
    for scf in scflist:
        for comp in scf.inputrange:
            if not comp in inputsets:
                inputsets += [comp]
                if len(inputsets) > 24:
                    raise BaseException('Too many NTs required')
    #print('NTs determined')
    #for iset in inputsets:
        #print([e.root for e in iset])
    CFG = []
    for scf in scflist:
        NTname = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z']
        for NT in NTname:
            if NT in scf.root:
                raise BaseException('Word contains character ' + NT + ', which is reserved for non-terminals')
        
        #print('make rules for '+scf.root)
        # make all rules for scf
        #determine category
        for h in range(len(categorization)):
            if scf in categorization[h]:
                cat = 'c'+str(h+1)
                break
        #determine contained NTs
        NTs = []
        for comp in scf.inputrange:
            for i in range(len(inputsets)):
                if comp == inputsets[i]:
                    NTs += [NTname[i]]
                    break
        #compose CFG word
        subwords = scf.root.split('_')
        CFGword = ''
        for j in range(len(NTs)):
            CFGword += subwords[j] + NTs[j]
        CFGword += subwords[-1]
        #make rules
        CFG += [('S',CFGword,cat,scf.mapping)]
        #print(CFG[-1])
        for k in range(len(inputsets)):
            #print('is '+scf.root+' in this inputset?')
            for entr in inputsets[k]:
                #entr.present()
                if entr.root == scf.root and entr.mapping == scf.mapping:
                    CFG += [(NTname[k],CFGword,cat,scf.mapping)]
                    #print(CFG[-1])
                    break
        #print(CFG)
    return CFG


# In[3]:


def cfg2mg(CFG, documentation = True):
    '''
    Converts a categorized CFG into an MG
    Input: Categorized CFG = List of of rules
    Rule = 3-tuple (Non-Terminal, Word, Category), 
    i.e. the Non-terminal can generate the word Word of category Category
    Capital letters are always interpretted as non-terminals here, minor letters as terminals.
    Words are strings out of terminals and non-terminals.
    Categories are strings that describe the type/category of a produced word.
    S is the default start symbol
    '''
    #for rule in CFG:
        #print(rule)
    CFG += [('START', 'S', 'cFin',[1,0])]
    N = set([rule[0] for rule in CFG]) # Set of all NTs
    F = [] # Free NTs
    R = [] # Restricted NTs
    target_category = {} # Dictionary that stores the target category of each NT
    
    category_order = determine_partial_category_order(CFG)
    if documentation:
        print('Category order:')
        print(category_order.relations)
    aux_NTs = []
    for NT in N:
        # adapt the grammar so that NT has one unique target category
        # and write this in the dictionary 'target_category'
        #copy_C = CFG[:]
        NT_rules = [rule for rule in CFG if rule[0] == NT]
        targetted_categories = [rule[2] for rule in NT_rules]
        if len(targetted_categories) > 1:
            #print('Rules of NT ' + NT + ': ')
            #print(NT_rules)
            main_category = category_order.maximum(targetted_categories, documentation)
            if documentation:
                print('main target category of '+NT+' is '+main_category)
            target_category[NT] = main_category
            updated_NT_rules = [rule for rule in NT_rules if rule[2] == main_category]
            secondary_categories = list(set(targetted_categories) - set([main_category])) 
            #print('secondary categries:')
            #print(secondary_categories)
            for ind in range(len(secondary_categories)):
                category_order.relations.add((main_category,secondary_categories[ind]))
                updated_NT_rules += [(NT, NT+str(ind), main_category, [1,0])]
                target_category[NT+str(ind)] = secondary_categories[ind]
                aux_NTs += [NT+str(ind)]
                for rule in NT_rules:
                    if rule[2] == secondary_categories[ind]:
                        updated_NT_rules += [(NT+str(ind), rule[1], secondary_categories[ind], rule[3])]
            #print('Updated rules:')
            #print(updated_NT_rules)
            CFG = [rule for rule in CFG if rule not in NT_rules] + updated_NT_rules
        else:
            target_category[NT] = targetted_categories[0]
        #print('Any changes to CFG?')
        #print(not copy_C == CFG)
        #print('Any changes to target categories?')
        #print(target_category)
        #print('Any changes to category order?')
        #print([rel for rel in category_order.relations])
        # decide if NT controls free input slots or restricted ones
        if free(NT,CFG):
            F += [NT]
        else:
            R += [NT]
    for n in aux_NTs:
        N.add(n)
        if free(n,CFG):
            F += [n]
        else:
            R += [n]
    #print('Free NTs:')
    #print(F)
    #print('Restricted NTs:')
    #print(R)
    #print('Target Categories:')
    #print(target_category)
    
    transformable_CFG = []
    for rule in CFG:
        # decompose each rule into rules that are transformable into MG items
        # it means that they may only have NTs at their first or last spot and they have to be one of the shapes
        # word, Rword, wordF or FwordF
        transformable_CFG += decompose_rule(rule,R,F,N,target_category)
    
    #print('Transformable CFG:')
    #for rule in transformable_CFG:
        if documentation:
            print(rule)
    #print('Free:')
    #print(F)    
    #print('Target Categories:')
    #print(target_category)

    summarized_tf_CFG = []
    for rule in transformable_CFG:
        same_rule_found = False
        for i in range(len(summarized_tf_CFG)):
            # check if 2 rules produce same string and category
            if rule[1].replace('*','') == summarized_tf_CFG[i][1].replace('*','') and rule[2] == summarized_tf_CFG[i][2]:
                same_rule_found = True
                # then: summarize them into one
                if not rule[0] in summarized_tf_CFG[i][0]:
                    summarized_tf_CFG[i][0] += [rule[0]]
        if not same_rule_found:
            # otherwise: add new rule
            summarized_tf_CFG += [[[rule[0]],rule[1],rule[2],rule[3]]]
    if documentation:
        print('Summarized transformed CFG:')
        for rule in summarized_tf_CFG:
            print(rule)
    
    MG_without_licensors = []
    for rule in summarized_tf_CFG:
        # create a proper MG item
        # created as a 3-tuple, so in the 2nd and 3rd entry information can be stored, based on which licensors are assigned later
        #print('model:')
        #print(rule)
        chars = list(rule[1])
        for char in reversed(range(len(chars))):
            if chars[char][0] == '*' or chars[char][0].isnumeric() or chars[char][0] == '+':
                chars = chars[:char - 1] + [chars[char - 1] + chars[char]] + chars[char + 1:]
        #print('characters:')
        #print(chars)
        # produce semantic string
        if len(rule[3]) == 1:
            sem = ', '+str(rule[3][-1])
        elif len(rule[3]) == 2:
            if rule[3] == [1,0]:
                sem = ', '+"\u03BB"+"x.x"
            elif rule[3][-1] == 0:
                sem = ', '+"\u03BB"+"x.'"+str(rule[3][0])+"X'(x)"
            else:
                sem = ', '+"\u03BB"+"x.'"+str(rule[3][0])+"X+"+str(rule[3][-1])+"'(x)"
        elif len(rule[3]) == 3:
            if rule[3][-1] == 0:
                sem = ', '+"\u03BB"+"x"+"\u03BB"+"y.'"+str(rule[3][1])+"X+"+str(rule[3][0])+"Y'(x,y)"
            else:
                sem = ', '+"\u03BB"+"x"+"\u03BB"+"y.'"+str(rule[3][1])+"X+"+str(rule[3][0])+"Y+"+str(rule[3][-1])+"'(x,y)"
        # compose full string
        if chars == []:
            item =('[ :: ' + rule[2] +']', rule[0], rule[2],rule[3],sem)
        elif chars[0] not in N and chars[-1] not in N: # if word
            #print('no NTs')
            item =('[' + ''.join(chars) + ' :: ' + rule[2] +']', rule[0], rule[2],rule[3],sem)
        elif chars[0] in R: # if Rword
            #print('left restricted')
            item = ('[' + ''.join(chars[1:]) + ' :: ' + '=' + target_category[chars[0]] + ', +' + chars[0] + ', ' + rule[2] +']', rule[0], rule[2],rule[3],sem)
        elif (chars[0] not in N or len(chars) == 1) and chars[-1] in F: # if wordF
            #print('right free')
            item =('[' + ''.join(chars[:-1]) + ' :: ' + '=' + target_category[chars[-1]] + ', ' + rule[2] +']', rule[0], rule[2],rule[3],sem)
        elif chars[0] in F and chars[-1] in F: # if FwordF
            #print('2-handed free')
            item =('[' + ''.join(chars[1:-1]) + ' :: ' + '=' + target_category[chars[-1]] + ', ' + '=' + target_category[chars[0]] + ', ' + rule[2] +']', rule[0], rule[2],rule[3],sem)
        #print('item:')
        #print(item)
        MG_without_licensors += [item]
        
    #remove items whose category has no selector in the lexicon
    useless_items = []
    for item in MG_without_licensors:
        if item[2] == 'cFin':
            useless = False
        else:
            useless = True
            for other_item in MG_without_licensors:
                if '='+item[2] in other_item[0] and other_item not in useless_items:
                    useless = False
                    break
        if useless:
            useless_items += [item]
    for item in useless_items:
        MG_without_licensors.remove(item)
    
    MG = []
    for item in MG_without_licensors:
        licensors = ['-'+NT for NT in item[1] if NT in R]
        #print('licensors of ' + item[0] + ':')
        #print(licensors)
        if licensors != []:
            new_item = (item[0][:-1] + ''.join([', ' + licensor for licensor in licensors[:-1]]) + ', ' + licensors[-1] + 's' + ']'+item[4], licensors, item[2])
        else:
            new_item = (item[0] +item[4], licensors, item[2])
        MG += [new_item]
    
    all_licensors = set()
    all_licensor_chains = set()
    
    for item in MG:
        for licensor in range(len(item[1])):
            if licensor == len(item[1]) - 1:
                all_licensors.add((item[1][licensor] + 's',item[2]))
            else:
                all_licensors.add((item[1][licensor],item[2]))
            if licensor == len(item[1]) - 1: #< len(item[1]): # - 1:
                all_licensor_chains.add((tuple([item[1][licensor]+'s']),item[2])) # + item[1][1:]
            else:
                all_licensor_chains.add((tuple(item[1][licensor:-1] + [item[1][-1]+'s']),item[2]))
    
    #print('Remove shapers:')
    for licensor in all_licensors:
        remove_shaper = ['[ :: =' + licensor[1] + ', +' + licensor[0][1:] + ', ' + licensor[1] + ']'+', '+"\u03BB"+"x.x"]
        #print(remove_shaper)
        MG += [remove_shaper]
        
    #print('Select shapers:')
    for chain in all_licensor_chains:
        #print('chain:')
        #print(chain)
        if chain[0][0][-1] == 's':
            select_shaper = ['[ :: =' + chain[1] + ', +' + chain[0][0][1:] #+ 's' 
                         + ''.join([', +' + licensor[1:] for licensor in chain[0][1:]]) + ', ' + chain[1] + ', ' +chain[0][0][:-1] + ']'', '+"\u03BB"+"x.x"]
        else:
            select_shaper = ['[ :: =' + chain[1] + ', +' + chain[0][0][1:] #+ 's' 
                         + ''.join([', +' + licensor[1:] for licensor in chain[0][1:]]) + ', ' + chain[1] + ', ' +chain[0][0] + ']'', '+"\u03BB"+"x.x"]
        #print(select_shaper)
        MG += [select_shaper]
        
    print('MG: ('+str(len(MG))+' items)')
    print('\n'.join([item[0] for item in MG]))

def vari(string):
    return string + "*"
def contr(string):
    return string + '+'

def decompose_rule(rule,R,F,N,target_category):
    '''
    Input: one rule
    Output: list of rules that the rule is decomposed into
    Assumptions: 
        N = list of all non-terminals
        F = list of all free non-terminals
        R = list of all restricted non-terminals
        vari(char) is a function that produces a uniquely indexed version of char
    '''
    symbol = rule[0]
    string = rule[1]
    mapping = rule[3]
    if type(string) == str:
        string = list(string)
        for char in reversed(range(len(string))):
            if string[char][0] == '*' or string[char][0].isnumeric():
                string = string[:char - 1] + [string[char - 1] + string[char]] + string[char + 1:]
    categ = rule[2]
    #print('decompose ' + str(string))
    for spot in range(1, len(string) - 1):
        char = string[spot]
        if char in N:
            aux_NT = vari(char)
            aux_cat = vari(categ)
            while aux_NT in N:
                aux_NT = vari(aux_NT)
            while aux_cat in target_category.values():
                aux_cat = vari(aux_cat)
            #print('decompose into:')
            #print(string[:spot]+[aux_NT])
            #print(string[spot:])
            N.add(aux_NT)
            F += [aux_NT]
            target_category[aux_NT] = aux_cat
            if string[0] in N:
                left_mapping = [mapping[0],1,mapping[-1]]
                right_mapping = mapping[1:-1]+[0]
            else:
                left_mapping = [1,mapping[-1]]
                right_mapping = mapping[0:-1]+[0]
            return decompose_rule([symbol,string[:spot]+[aux_NT],categ,left_mapping],R,F,N,target_category) + decompose_rule([aux_NT,string[spot:],aux_cat,right_mapping],R,F,N,target_category)
    if len(string) > 1 and string[0] in R and string[-1] in R:
        #print('2-handed restricted')
        F += [contr(string[0])]
        N.add(contr(string[0]))
        target_category[contr(string[0])] = categ + string[0]
        F += [contr(string[-1])]
        N.add(contr(string[-1]))
        target_category[contr(string[-1])] = categ + string[-1]
        return [[symbol,contr(string[0])+''.join(string[1:-1])+contr(string[-1]),categ,mapping],[contr(string[0]),string[0],categ + string[0],[1,0]],[contr(string[-1]),string[-1],categ + string[-1],[1,0]]]
    elif len(string) > 1 and string[-1] in R:
        #print('right restricted')
        F += [contr(string[-1])]
        N.add(contr(string[-1]))
        target_category[contr(string[-1])] = categ + string[-1]
        return [[symbol,''.join(string[:-1])+contr(string[-1]),categ,mapping],[contr(string[-1]),string[-1],categ + string[-1],[1,0]]]
    elif len(string) > 1 and string[0] in R and string[-1] in F:
        #print('right free, left restricted')
        F += [contr(string[0])]
        N.add(contr(string[0]))
        target_category[contr(string[0])] = categ + string[0]
        return [[symbol,contr(string[0])+''.join(string[1:]),categ,mapping],[contr(string[0]),string[0],categ + string[0],[1,0]]]
    elif len(string) > 1 and string[0] in F and string[-1] not in N:
        #print('left free only')
        F += ['O']
        N.add('O')
        target_category['O'] = 'cnix'
        return [[symbol,''.join(string)+'O',categ,mapping[:-1]+[1]+[mapping[-1]]],['O','','cnix',[0]]]
    else:
        #print('No more decomposing needed')
        return [[rule[0],''.join(rule[1]),rule[2],mapping]]
    
def free(NT,CFG):
    NT_rules = [NT_rule for NT_rule in CFG if NT_rule[0] == NT]
    for rule in CFG:
        # check that rule is not of NT but shares a common category with NT
        if rule not in NT_rules and rule[2] in [NT_rule[2] for NT_rule in NT_rules]:
            NT_rules_of_same_category = [NT_rule for NT_rule in NT_rules if NT_rule[2] == rule[2]]
            # check if in the category rule produces a word that NT does not
            if rule[1] not in [NT_rule[1] for NT_rule in NT_rules_of_same_category]:
                #print(NT+' is not free because of '+str(rule))
                # then NT would be restricted
                return False
    return True
    '''for NT_rule in CFG:
        if NT_rule[0] == NT:
            for rule in CFG:
                if rule[0] != NT and rule[1] != NT_rule[1] and rule[2] == NT_rule[2]:
                    return False
    return True'''

def restricted(NT,CFG):
    return not free(NT,CFG)
     
        
def unify_target_category(NT,CFG,target_category,category_order):
    copy_CFG = CFG[:]
    NT_rules = [rule for rule in copy_CFG if rule[0] == NT]
    targetted_categories = [rule[2] for rule in NT_rules]
    if len(targetted_categories) > 1:
        main_category = category_order.maximum(targetted_categories)
        #print('main target category of '+NT+' is '+main_category)
        target_category[NT] = main_category
        updated_NT_rules = [rule for rule in NT_rules if rule[2] == main_category]
        secondary_categories = list(set(targetted_categories) - set([main_category])) 
        #print('secondary categries:')
        #print(secondary_categories)
        for ind in range(len(secondary_categories)):
            updated_NT_rules += [(NT, NT+str(ind), main_category)]
            for rule in NT_rules:
                if rule[2] == secondary_categories[ind]:
                    updated_NT_rules += [(NT+str(ind), rule[1], secondary_categories[ind])]
        #print('New rules:')
        #print(updated_NT_rules)
        copy_CFG = [rule for rule in copy_CFG if rule not in NT_rules] + updated_NT_rules
    else:
        target_category[NT] = targetted_categories[0]
    CFG = copy_CFG[:]
    
    
def determine_partial_category_order(CFG):
    N = set([rule[0] for rule in CFG])
    category_order = partial_order([])
    for rule in CFG:
        for char in rule[1]:
            if char in N:
                for rulec in CFG:
                    if rulec[0] == char:
                        category_order.relations.add((rule[2],rulec[2]))
    return category_order

class partial_order:
    def __init__(self, relations):
        self.relations = set(relations)

    def compare(self, a, b, documentation=False):
        if documentation:
            print(a + ' < ',end='')
        if (a, b) in self.relations:
            if documentation:
                print(b)
            return True
        for r in self.relations:
            if r[0] == a and self.compare(r[1], b, documentation):
                return True
        return False
    
    def add_rule(rule):
        self.relations.add(rule)

    def maximum(self, lst, documentation = False):
        max_element = None
        for element in lst:
            if not max_element or self.compare(element, max_element, documentation):
                max_element = element
        return max_element


# In[ ]:


if __name__ == '__main__':
    print('Hello, I am a chatbot who would like to learn the numeral words in your language.') 
    print('If you help me, I can write you a context-free grammar and a minimalist grammar that can generate exactly all the numeral words in your language.')
    print('To start the program, write a number up to which you want to teach me the numeral words.') 
    print('You can shorten the learning process later by typing to STOP at any point.')
    if (grenze := input('\n')).isnumeric():
        #print('Dann los')
        print('I will now ask you about the numeral words in your language in ascending order. I will decompose the numeral words and if I recognize a pattern in the breakdowns, I will try to derive numeral words myself')
        print('')
        learnerlex = []
        for number in range(1,int(grenze)):
            interrupt = False
            if not any([number in [outp.mapping[-1] for outp in entr.all_outputs()] for entr in learnerlex]):
                print('How do you write '+str(number)+'?')
                while (word := input('\n')) != 'X':
                    if word == 'STOP':
                        interrupt = True
                        break
                    parse = advanced_parse(number, word, learnerlex, False, True)
                    understood = False
                    for entry in learnerlex:
                        if entry.root == parse.root:
                            understood = True
                            print('This is related to '+entry.root)
                            learned = entry.merge(parse)
                            learnerlex.remove(entry)
                            if entry.actual_dimension() < learned.actual_dimension():
                                print('I think I understand more now')
                                #learned = learned.reinforce()
                                copy = learned
                                candidates_for_abstraction = []
                                upper_limit = sum([max([0,coeff]) for coeff in learned.mapping]) - 1
                                for entry2 in learnerlex:
                                    if all([fe.mapping[-1] < upper_limit for fe in entry2.all_outputs()]):
                                        candidates_for_abstraction += [entry2]
                                invariant_slots = []
                                for comp in range(copy.dimension()):
                                    if len(copy.inputrange[comp]) == 1:
                                        invariant_slots += [comp]
                                for combination in cartesian_product((copy.dimension() - len(invariant_slots)) * [candidates_for_abstraction]):
                                    cand = []
                                    for comp in range(copy.dimension()):
                                        if comp in invariant_slots:
                                            cand += [copy.inputrange[comp][0]]
                                        else:
                                            cand += [combination[0]]
                                            combination = combination[1:]
                                    word_cand = copy.insert([candc.sample() for candc in cand])
                                    #print(word_cand.root)
                                    proposed_inputrange = []
                                    entry_is_new = False
                                    for comp in range(copy.dimension()):
                                        if cand[comp].root in [entr.root for entr in copy.inputrange[comp]]:
                                            proposed_inputrange += [copy.inputrange[comp]]
                                        else:
                                            proposed_inputrange += [copy.inputrange[comp]+[cand[comp]]]
                                            entry_is_new = True
                                    if entry_is_new:
                                        proposal = SCFunction(copy.root,proposed_inputrange,copy.mapping)
                                        proposal_is_new = True
                                        for entr in learnerlex:
                                            for word in entr.all_outputs():
                                                if word_cand.root == word.root or word_cand.mapping[-1] == word.mapping[-1]:
                                                    #print('proposal ' + word_cand.root + ' ' + str(word_cand.mapping[-1]) + ' already covered by ' + word.root + ' ' + str(word.mapping[-1]))
                                                    proposal_is_new = False
                                                    break
                                            if not proposal_is_new:
                                                break
                                        if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                                            print('Is there also a numeral word '+ word_cand.root + '? (y/n)')
                                            #while (answer2 := input('\n')) not in ['ja','nein']:
                                            if True:
                                                if (answer2 := input('\n')) == 'y':
                                                    cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,learnerlex,False,False)
                                                    if cand_parse.root == learned.root:                                
                                                        copy = proposal
                                                        copy.present()
                                                    else:
                                                        pass
                                                        print('OK, but I think this is not related')
                                                if answer2 == 'n':
                                                    pass
                                                if answer2 == 'STOP':
                                                    interrupt = True
                                                    break
                                                
                                learned = copy   
                                
                    if not understood:
                        print('I remember that')
                        learned = parse
                    learned.present()
                    learnerlex += [learned]
                    if number % 1 == 0:
                        print('My lexicon now consists of:')
                        print([e.root for e in learnerlex])
                    break
                if interrupt:
                    break
        print('')
        print('Learning process completed. Now I generate a CFG for the learned numeral words')
        cfg=scflist2cfg(learnerlex)
        for rule in cfg:
            print(rule)
        print('')
        print('And now I convert the CFG into an MG.')
        cfg2mg(cfg)
        


# In[ ]:





# In[ ]:




