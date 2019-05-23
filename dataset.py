import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from stanza.nlp.corenlp import CoreNLPClient


client = None

# do not change
ACTIVATED_DOMAINS = ["hotel", "train", "restaurant", "attraction", "taxi"]


def annotate(sent):
    # global client
    # if client is None:
    #     client = CoreNLPClient(default_annotators='ssplit,tokenize'.split(','))
    # words = []
    # for sent in client.annotate(sent).sentences:
    #     for tok in sent:
    #         words.append(tok.word)
    words = sent.split()
    return words


class Turn:

    def __init__(self, turn_id, transcript, turn_label, belief_state, system_acts, system_transcript, num=None):
        self.id = turn_id
        self.transcript = transcript
        self.turn_label = turn_label
        self.belief_state = belief_state
        self.system_acts = system_acts
        self.system_transcript = system_transcript
        self.num = num or {}

    def to_dict(self):
        return {'turn_id': self.id, 'transcript': self.transcript, 'turn_label': self.turn_label, 'belief_state': self.belief_state, 'system_acts': self.system_acts, 'system_transcript': self.system_transcript, 'num': self.num}

    @classmethod
    def from_dict(cls, alld, i, d):
        # if i!=0:
        #     d["transcript"] = alld[i-1]["transcript"] + [";"] + d["transcript"]
        return cls(**d)

    @classmethod
    def annotate_raw(cls, raw, only_domain=""):
        system_acts = []
        for a in raw['system_acts']:
            if isinstance(a, list):
                s, v = a
                system_acts.append(['inform'] + s.split() + ['='] + v.split())
            else:
                system_acts.append(['request'] + a.split())
        # NOTE: fix inconsistencies in data label
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        if only_domain!="":
            return cls(
                turn_id=raw['turn_idx'],
                transcript=annotate(raw['transcript']),
                system_acts=system_acts,
                turn_label=[[fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())] for s, v in raw['turn_label'] if only_domain in s],
                belief_state=[bs for bs in raw['belief_state'] if only_domain in bs["slots"][0][0]],
                system_transcript=annotate(raw['system_transcript']),
            )
        else:
            return cls(
                turn_id=raw['turn_idx'],
                transcript=annotate(raw['transcript']),
                system_acts=system_acts,
                turn_label=[[fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())] for s, v in raw['turn_label'] if s.split("-")[0] in ACTIVATED_DOMAINS],
                belief_state=[bs for bs in raw['belief_state'] if bs["slots"][0][0].split("-")[0] in ACTIVATED_DOMAINS], #raw['belief_state'],
                system_transcript=annotate(raw['system_transcript']),
            )

    def numericalize_(self, vocab):
        self.num['transcript'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.transcript + ['<eos>']], train=True)
        self.num['system_transcript'] = vocab.word2index(['<sos>'] + [w.lower() for w in self.system_transcript + ['<eos>']], train=True)
        self.num['system_acts'] = [vocab.word2index(['<sos>'] + [w.lower() for w in a] + ['<eos>'], train=True) for a in self.system_acts + [['<sentinel>']]]


class Dialogue:

    def __init__(self, dialogue_id, turns):
        self.id = dialogue_id
        self.turns = turns

    def __len__(self):
        return len(self.turns)

    def to_dict(self):
        return {'dialogue_id': self.id, 'turns': [t.to_dict() for t in self.turns]}

    @classmethod
    def from_dict(cls, d):
        return cls(d['dialogue_id'], [Turn.from_dict(d['turns'], i, t) for i, t in enumerate(d['turns'])])
        # return cls(d['dialogue_id'], [Turn.from_dict(t) for t in d['turns']])

    @classmethod
    def annotate_raw(cls, raw, only_domain=""):
        return cls(raw['dialogue_idx'], [Turn.annotate_raw(t, only_domain) for t in raw['dialogue']])


class Dataset:

    def __init__(self, dialogues):
        self.dialogues = dialogues

    def __len__(self):
        return len(self.dialogues)

    def iter_turns(self):
        for d in self.dialogues:
            for t in d.turns:
                yield t

    def to_dict(self):
        return {'dialogues': [d.to_dict() for d in self.dialogues]}

    @classmethod
    def from_dict(cls, d):
        return cls([Dialogue.from_dict(dd) for dd in d['dialogues']])

    @classmethod
    def annotate_raw(cls, fname, only_domain=""):
        with open(fname) as f:
            data = json.load(f)
            if only_domain!="":
                return cls([Dialogue.annotate_raw(d, only_domain) for d in tqdm(data) if only_domain in d["domains"]])
            else:
                return cls([Dialogue.annotate_raw(d) for d in tqdm(data)])

    def numericalize_(self, vocab):
        for t in self.iter_turns():
            t.numericalize_(vocab)

    def extract_ontology(self):
        slots = set()
        values = defaultdict(set)
        for t in self.iter_turns():
            for s, v in t.turn_label:
                slots.add(s.lower())
                values[s].add(v.lower())
        return Ontology(sorted(list(slots)), {k: sorted(list(v)) for k, v in values.items()})

    def batch(self, batch_size, shuffle=False):
        turns = list(self.iter_turns())
        if shuffle:
            np.random.shuffle(turns)
        for i in tqdm(range(0, len(turns), batch_size)):
            yield turns[i:i+batch_size]

    def evaluate_preds(self, preds, Ontology=[]):
        request = []
        inform = []
        joint_goal = []
        slot_acc = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number'}
        i = 0
        print("len of Ontology", len(Ontology.slots))
        for d in self.dialogues:
            pred_state = {}
            for t in d.turns:
                gold_request = set([(s, v) for s, v in t.turn_label if s == 'request'])
                gold_inform = set([(s, v) for s, v in t.turn_label if s != 'request'])
                pred_request = set([(s, v) for s, v in preds[i] if s == 'request'])
                pred_inform = set([(s, v) for s, v in preds[i] if s != 'request'])
                request.append(gold_request == pred_request)
                inform.append(gold_inform == pred_inform)

                slot_acc.append(self.compute_acc(gold_inform, pred_inform, len(Ontology.slots)))

                gold_recovered = set()
                pred_recovered = set()
                for s, v in pred_inform:
                    pred_state[s] = v
                for b in t.belief_state:
                    for s, v in b['slots']:
                        if b['act'] != 'request':
                            gold_recovered.add((b['act'], fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())))
                for s, v in pred_state.items():
                    pred_recovered.add(('inform', s, v))
                joint_goal.append(gold_recovered == pred_recovered)
                i += 1
        return {'turn_inform': np.mean(inform), 'turn_request': np.mean(request), 'joint_goal': np.mean(joint_goal), 'slot_acc': np.mean(slot_acc)}

    def compute_acc(self, gold, pred, nb_slot):
        miss_gold = 0
        miss_slot = []

        # print(gold)
        # print(pred)
        # print(nb_slot)
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = nb_slot
        ACC = nb_slot - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def record_preds(self, preds, to_file):
        data = self.to_dict()
        i = 0
        for d in data['dialogues']:
            for t in d['turns']:
                t['pred'] = sorted(list(preds[i]))
                i += 1
        with open(to_file, 'wt') as f:
            json.dump(data, f)


class Ontology:

    def __init__(self, slots=None, values=None, num=None):
        self.slots = slots or []
        self.values = values or {}
        self.num = num or {}

    def __add__(self, another):
        new_slots = sorted(list(set(self.slots + another.slots)))
        new_values = {s: sorted(list(set(self.values.get(s, []) + another.values.get(s, [])))) for s in new_slots}
        return Ontology(new_slots, new_values)

    def __radd__(self, another):
        return self if another == 0 else self.__add__(another)

    def to_dict(self):
        return {'slots': self.slots, 'values': self.values, 'num': self.num}

    def numericalize_(self, vocab):
        self.num = {}
        for s, vs in self.values.items():
            self.num[s] = [vocab.word2index(annotate('{} = {}'.format(s, v)) + ['<eos>'], train=True) for v in vs]

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
