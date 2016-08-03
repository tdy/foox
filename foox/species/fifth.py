import random
from math import copysign
from collections import namedtuple
import foox.ga as ga
from utils import make_generate_function

# Some sane defaults.
DEFAULT_POPULATION_SIZE = 400
DEFAULT_MAX_GENERATION = 200
DEFAULT_MUTATION_RANGE = 3
DEFAULT_MUTATION_RATE = 0.3

HSCORE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 2.0,
                  1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0]

VSCORE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0, 1.0,
                  1.0, 1.0, 1.0, 1.0]

# Intervals between notes that are allowed in fifth species counterpoint.
CONSONANCES = [2, 4, 5, 7, 9, 11]
DISSONANCES = [3, 6, 8, 10]
PERFECT_INTERVALS = [0, 3, 4, 7]


def set_weights(hweights, vweights):
    global HSCORE_WEIGHTS
    global VSCORE_WEIGHTS
    HSCORE_WEIGHTS = hweights
    VSCORE_WEIGHTS = vweights

def generate_rhythmic_pattern(cantus_firmus):
    # Cantus firmus should be longer than 3 measures
    if len(cantus_firmus) < 3:
        return list()
    # Pattern format: [[duration, is_tied_forward], ...]
    # Duration: 8 = whole note, 4 = half note, 2 = quarter note, 1 = eighth note
    basic_patterns = [
       [[4, False], [4, False]],  # base 4 4
       [[2, False], [2, False], [2, False], [2, False]],  # base 2 2 2 2
       [[4, False], [2, False], [2, False]],  # base 4 2 2
       [[4, False], [4, True]],  # base 4 4
       [[2, False], [2, False], [4, True]],  # base 2 2 4
       [[4, False], [2, False], [1, False], [1, False]],  # base 4 2 2
       [[2, False], [1, False], [1, False], [2, False], [2, False]],  # base 2 2 2 2
       [[2, False], [2, False], [2, False], [1, False], [1, False]],  # base 2 2 2 2
       [[2, False], [1, False], [1, False], [4, True]]  # base 2 2 4
    ]
    # Replacing quarter with two eights or adding a tie are not considered to be a significant rhythmic changes
    # We need to consider it when preventing repetitiveness
    pattern_bases = [0, 1, 2, 0, 3, 2, 1, 1, 3]
    pattern_sequence = list()

    # First measure (half pause and half note)
    if len(cantus_firmus) == 3:
        pattern_sequence.append(3)
    else:
        pattern_sequence.append(random.choice([0, 3]))

    for i in range(1, len(cantus_firmus) - 2):
        # For measure before penultimate last note is tied
        cur_pattern = random.randint(0, 8) if i != len(cantus_firmus) - 3 else random.choice([3, 4, 8])
        prev1_base = pattern_bases[pattern_sequence[-1]]
        prev2_base = pattern_bases[pattern_sequence[-2]] if i > 1 else -1
        # For bases 0, 1 (notes of similar duration) no more then 2 similar measures in the row
        # For bases 2, 3 no similar measures in the row at all
        if prev1_base in [2, 3] or prev1_base == prev2_base:
            while pattern_bases[cur_pattern] == prev1_base:
                # For measure before penultimate last note is tied
                cur_pattern = random.randint(0, 8) if i != len(cantus_firmus) - 3 else random.choice([3, 4, 8])
        pattern_sequence.append(cur_pattern)

    rhythmic_pattern = list()
    for pattern_num in pattern_sequence:
        rhythmic_pattern += basic_patterns[pattern_num]
    # Penultimate measure
    rhythmic_pattern += [[2, False], [1, False], [1, False], [4, False]]
    # Last measure
    rhythmic_pattern.append([8, False])
    durations = list()
    # Convert rhythmic pattern to the list of note durations
    print rhythmic_pattern
    for i in range(len(rhythmic_pattern)):
        if i == 0 or not rhythmic_pattern[i - 1][1]:
            durations.append(rhythmic_pattern[i][0])
        else:
            durations[-1] += rhythmic_pattern[i][0]
    return durations

def create_population(number, cantus_firmus):
    durations = generate_rhythmic_pattern(cantus_firmus)
    result = []
    for i in range(number):
        current_beat = 0
        cf_note_idx = 0
        new_chromosome = []
        for d, duration in enumerate(durations):
            cf_note = cantus_firmus[cf_note_idx]
            cp_note = cantus_firmus[cf_note_idx]
            if d == 0:
                # Start with half pause
                cp_note = 17
            elif d == 1:
                valid_range = [interval for interval in [0, 5, 7] if (interval + cf_note) < 17]
                cp_note += random.choice(valid_range)
            elif d == len(durations) - 5:
                valid_range = [interval for interval in DISSONANCES if (interval + cf_note) < 17 and
                               cantus_firmus[-2] - cantus_firmus[-3] + interval in CONSONANCES]
                if len(valid_range) == 0:
                    valid_range = [interval for interval in range(1, 12) if (interval + cf_note) < 17 and
                                   cantus_firmus[-2] - cantus_firmus[-3] + interval in CONSONANCES]
                cp_note += random.choice(valid_range)
            elif d == len(durations) - 1:
                valid_range = [interval for interval in [0, 7] if (interval + cf_note) < 17]
                cp_note += random.choice(valid_range)
            elif current_beat == 0:
                # Its better to keep first beat consonant.
                valid_range = [interval for interval in CONSONANCES if (interval + cf_note) < 17]
                cp_note += random.choice(valid_range)
            elif duration == 1 or new_chromosome[-1][0] == 1:
                # Eight notes move in step
                valid_range = list()
                if new_chromosome[-1][0] + 1 < 17:
                    valid_range.append(1)
                if new_chromosome[-1][0] - 1 > cf_note:
                    valid_range.append(-1)
                cp_note = new_chromosome[-1][0] + random.choice(valid_range)
            else:
                valid_range = [interval for interval in range(2, 12) if (interval + cf_note) < 17]
                cp_note += random.choice(valid_range)
            new_chromosome.append([cp_note, duration])
            current_beat += duration
            cf_note_idx += current_beat / 8
            current_beat %= 8
        new_chromosome[-1][0] = cantus_firmus[-1] + random.choice([0, 7])
        genome = Genome(new_chromosome)
        result.append(genome)
    return result


def generate_random(cantus_firmus):
    durations = generate_rhythmic_pattern(cantus_firmus)
    current_beat = 0
    cf_note_idx = 0
    random_cp = []
    for d, duration in enumerate(durations):
        cf_note = cantus_firmus[cf_note_idx]
        cp_note = cantus_firmus[cf_note_idx]
        valid_range = [interval for interval in range(1, 12) if (interval + cf_note) < 17]
        cp_note += random.choice(valid_range)
        random_cp.append([cp_note, duration])
        current_beat += duration
        cf_note_idx += current_beat / 8
        current_beat %= 8
    return random_cp



def make_fitness_function(cantus_firmus):
    """
    Given the cantus firmus, will return a function that takes a single Genome
    instance and returns a fitness score.
    """
    def fitness_function(genome, debug=False):
        """
        Given a candidate solution will return its fitness score assuming
        the cantus_firmus in this closure. Caches the fitness score in the
        genome.
        """
        cp = genome.chromosome

        # calculate some information for easier scoring
        ScoringInfo = namedtuple("ScoringInfo", "pitch duration measure beat vi hi_next hi_prev voice_dir")
        melody_info = list()
        beat = 4
        measure = 0
        for i in range(1, len(cp)):
            hi_next = abs(cp[i][0] - cp[i + 1][0]) if i != len(cp) - 1 else -1  # next horizontal interval
            hi_prev = abs(cp[i][0] - cp[i - 1][0]) if i != 1 else -1  # previous horizontal interval
            vi = abs(cp[i][0] - cantus_firmus[measure])  # vertical interval
            # voice movement direction
            voice_dir = 0 if i == len(cp) - 1 or cp[i + 1][0] == cp[i][0] else copysign(1, cp[i + 1][0] - cp[i][0])
            melody_info.append(ScoringInfo(cp[i][0], cp[i][1], measure, beat, vi, hi_next, hi_prev, voice_dir))
            beat += cp[i][1]
            measure += beat / 8
            beat %= 8

        if debug:
            print "MELODY INFO: ", melody_info

        hscores = list()
        vscores = list()
        # hscore 1: 8th notes must move in step
        amount_of_8th = 0
        amount_of_missteps = 0
        for note in melody_info:
            if note.duration == 1:
                amount_of_8th += 1
                if note.hi_next > 1:
                    amount_of_missteps += 1
                if note.hi_prev > 1:
                    amount_of_missteps += 1
        hscores.append(float(amount_of_missteps) / (amount_of_8th * 2))
        if debug:
            print "HSCORE 1: 8TH - ", amount_of_8th, ", MISSTEPS - ", amount_of_missteps

        # hscore 2: one climax, that can be repeated only after neighboring tone
        # hscore 3: Climax should be on the strong beat
        highest_note = max([note.pitch for note in melody_info])
        climax_count = 0
        climax_on_weak_beat_count = 0
        for i, note in enumerate(melody_info):
            if note.pitch == highest_note:
                climax_count += 1
                if note.beat not in [0, 4]:
                    climax_on_weak_beat_count += 1
                if i < len(melody_info) - 2 and note.pitch == melody_info[i + 2].pitch:      # If next note is
                    if note.hi_next == 1 and melody_info[i + 2].hi_prev == 1:                # neighboring tone
                        if note.vi in CONSONANCES and melody_info[i + 2].vi in CONSONANCES:  # And surrounding notes are consonant
                            climax_count -= 1                                                # we can allow 2nd climax
                            if melody_info[i + 2].beat not in [0, 4]:
                                climax_on_weak_beat_count -= 1  # And 2nd climax may be on weak beat

        hscores.append(float(climax_count - 1) / len(melody_info))
        hscores.append(float(climax_on_weak_beat_count) / climax_count)

        if debug:
            print "HSCORE 2+3: CLIMAX CNT - ", climax_count, ", WEAK CLIMAX CNT - ", climax_on_weak_beat_count

        # hscore 4: Horizontal intervals are consonant
        unconsonant_amount = len(filter(lambda x: x.hi_next not in CONSONANCES + [1], melody_info[:-1]))
        hscores.append(float(unconsonant_amount) / (len(melody_info) - 1))

        if debug:
            print "HSCORE 4: UNCONSANANT AMOUNT - ", unconsonant_amount

        # hscore 5: Stepwise movement should predominate
        leaps_count = len(filter(lambda x: x.hi_next != 1, melody_info[:-1]))
        sections = round(float(len(cantus_firmus)) / 16)
        if leaps_count < (2 * sections):
            hscores.append(float(leaps_count) / (len(melody_info) - 1 - 4 * sections))
        elif leaps_count > (4 * sections):
            hscores.append(float(leaps_count) / (len(melody_info) - 1 - 4 * sections))
        else:
            hscores.append(0.0)

        if debug:
            print "HSCORE 5: LEAPS - ", leaps_count, "SECTIONS - ", sections

        # hscore 6: After large leap  - stepwise motion
        large_leaps_count = 0
        large_leaps_not_followed_count = 0
        for i, note in enumerate(melody_info[:-1]):
            if note.hi_next >= 3:
                large_leaps_count += 1
                if melody_info[i + 1].hi_next != 1:
                    large_leaps_not_followed_count += 1
        hscores.append(float(large_leaps_not_followed_count) / large_leaps_count if large_leaps_count != 0 else 0.0)

        if debug:
            print "HSCORE 6: LL CNT - ", large_leaps_count, "LL NOT FOLLOWED CNT - ", large_leaps_not_followed_count

        # hscore 7: change direction after each large leap
        large_leaps_not_changedir_count = 0
        for i, note in enumerate(melody_info[:-1]):
            if note.hi_next >= 3 and note.voice_dir != -melody_info[i + 1].voice_dir:
                large_leaps_not_changedir_count += 1
        hscores.append(float(large_leaps_not_changedir_count) / large_leaps_count if large_leaps_count != 0 else 0.0)

        if debug:
            print "HSCORE 7: LL NOT CHNGDIR CNT - ", large_leaps_not_changedir_count

        # hscore 8: climax should be melodically consonant with tonic
        hscores.append(1.0 if highest_note - 4 in CONSONANCES else 0.0)

        # hscore 9: no more than 2 consecutive leaps
        conseq_leaps = 0
        punish_score = 0
        for note in melody_info:
            conseq_leaps += 1
            if note.hi_next in [0, 1]:
                conseq_leaps = 0
            if conseq_leaps > 3:
                punish_score += 1
        hscores.append(float(punish_score) / (len(melody_info) - 3))

        if debug:
            print "HSCORE 9: CONSEQ LEAPS PUNISH SCORE - ", punish_score

        # hscore 10: no more than 2 large leaps per section
        if large_leaps_count > 2 * sections:
            hscores.append(float(large_leaps_count - 2 * sections) / (len(melody_info) - 1 - 2 * sections))
        else:
            hscores.append(0.0)

        # hscore 11: not too long stepwise in same direction
        longest_stepwise_seq = 0
        current_stepwise_seq = 0
        prev_dir = 0
        num_changes = 0
        motion_vector = list()
        for note in melody_info:
            if note.hi_next <= 1:
                if note.voice_dir in [prev_dir, 0]:
                    current_stepwise_seq += 1
                    longest_stepwise_seq = max(longest_stepwise_seq, current_stepwise_seq)
                else:
                    prev_dir = note.voice_dir
                    current_stepwise_seq = 0
                    num_changes += 1
                    motion_vector.append(note.pitch)
            else:
                if note.voice_dir != prev_dir and note.voice_dir != 0:
                    prev_dir = note.voice_dir
                    num_changes += 1
                    motion_vector.append(note.pitch)
                current_stepwise_seq = 0
        motion_vector.append(cp[-1][0])
        if longest_stepwise_seq < 5:
            longest_stepwise_seq = 0
        hscores.append(float(longest_stepwise_seq) / len(cp))

        if debug:
            print "HSCORE 11: LONGEST STEPWISE SEQUENCE - ", longest_stepwise_seq

        # hscore 12: direction needs to change several times
        if num_changes < 3 * sections:
            hscores.append(1 - float(num_changes) / (3 * sections))
        else:
            hscores.append(0.0)

        # hscore 13: ending note is tonic
        hscores.append(0)

        # hscore 14: penultimate note is leading tone
        hscores.append(0)

        # hscore 15: the start of a motion is consonant with the end of a motion
        unconsotant_count = 0
        big_leaps_count = 0
        for i in range(1, len(motion_vector) - 1):
            if abs(motion_vector[i] - motion_vector[i + 1]) not in CONSONANCES:
                unconsotant_count += 1
            if abs(motion_vector[i] - motion_vector[i + 1]) > 6:
                big_leaps_count += 1
        hscores.append(float(unconsotant_count) / len(motion_vector))

        if debug:
            print "HSCORE 15: UNCONSONANT MOTIONS - ", unconsotant_count

        # hscore 16: Large motion intervals (>6 tones) should be avoided
        hscores.append(float(big_leaps_count) / len(motion_vector))

        if debug:
            print "HSCORE 16: LARGE MOTIONS - ", big_leaps_count

        # hscore 17: No frequent repetition of the same note
        rep_count = 0
        for note in melody_info:
            if note.hi_next == 0:
                rep_count += 1
        if rep_count > 2 * sections:
            rep_count -= 2 * sections
        else:
            rep_count = 0
        hscores.append(float(rep_count) / (len(cp) - 2 * sections))

        if debug:
            print "HSCORE 17: REPETITIONS COUNT - ", rep_count

        # hscore 18: no repetition of sequence within a 4 measure interval
        repeated = set()
        for i in range(len(melody_info) - 2):
            j = i + 1
            while melody_info[j].measure < melody_info[i].measure + 4 and j < len(melody_info) - 1:
                if melody_info[i].pitch == melody_info[j].pitch:
                    k = 1
                    while j + k < len(melody_info) and melody_info[j + k].pitch == melody_info[i + k].pitch:
                        if k == 1:
                            repeated.add(j)
                        repeated.add(j + k)
                        k += 1
                j += 1

        hscores.append(float(len(repeated)) / len(cp))

        if debug:
            print "HSCORE 18: REPEATED POSITIONS - ", repeated

        # hscore 19: largest allowed interval is octave
        more_than_ocatave_amount = len(filter(lambda x: x.hi_next > 7, melody_info[:-1]))
        hscores.append(float(more_than_ocatave_amount) / len(cp))

        if debug:
            print "HSCORE 19: MORE THAN OCTAVES - ", more_than_ocatave_amount

        # vscore 1: whole notes should be consonant (ensured by generation and hscore 13)
        vscores.append(0.0)

        # vscores 2 and 3: halves and quarters should be consonant on first beat.
        # or can be dissonant on other beats beat, if passing tone
        amount_of_notes = 0
        amount_of_wrong_notes = 0
        for i, note in enumerate(melody_info):
            if note.duration >= 2:
                amount_of_notes += 1
                if note.beat == 0 and note.vi not in CONSONANCES:
                    amount_of_wrong_notes += 1
                if note.beat != 0 and note.vi not in CONSONANCES:
                    amount_of_wrong_notes += 1
                    if note.hi_prev == 1 and note.hi_next == 1 and note.voice_dir == melody_info[i - 1].voice_dir:
                        if melody_info[i - 1].vi in CONSONANCES and melody_info[i + 1].vi in CONSONANCES:
                            amount_of_wrong_notes -= 1

        vscores.append(float(amount_of_wrong_notes) / amount_of_notes)
        vscores.append(float(amount_of_wrong_notes) / amount_of_notes)

        if debug:
            print "VSCORE 2+3: NOTES > THAN 8TH - ", amount_of_notes, ",DISSONANT ONES - ", amount_of_wrong_notes

        # vscore 4: one of eight notes from pair should be consonant
        amount_of_wrong_notes = 0
        for i, note in enumerate(melody_info[:-1]):
            if note.duration == 1 and melody_info[i + 1].duration == 1:
                if note.vi not in CONSONANCES and melody_info[i + 1].vi not in CONSONANCES:
                        amount_of_wrong_notes += 1
            beat += cp[i][1]
        vscores.append(float(amount_of_wrong_notes) / amount_of_8th)

        if debug:
            print "VSCORE 4: 8TH NOTES - ", amount_of_8th, ",DISSONANT ONES - ", amount_of_wrong_notes

        # vscore 5: unisons ok if on 1st beat through suspension or if tied over (ensured by storing format)
        # else: if followed by step

        wrong_unsiones = len(filter(lambda x: x.vi == 0 and x.hi_next != 1, melody_info))
        vscores.append(float(wrong_unsiones) / len(cp))

        if debug:
            print "VSCORE 5: WRONG UNISONES - ", wrong_unsiones

        # vscore 6: max allowed interval between voices is 10th, except for climax
        big_vert_intervals = len(filter(lambda x: x.vi > 9 and x.pitch != highest_note, melody_info))
        vscores.append(float(big_vert_intervals) / len(melody_info))

        if debug:
            print "VSCORE 6: VERT INTERVALS > 10TH - ", big_vert_intervals


        # vscore 7: There should be no crossing (ensured by generation)
        vscores.append(0.0)

        # vscore 8: avoid the overlapping of parts (ensured by generation)
        vscores.append(0.0)

        # vscore 9: no leaps from unison to octave and vice versa
        uni_to_oct_count = 0
        for i, note in enumerate(melody_info[:-1]):
            if note.vi == 7 and melody_info[i + 1].vi == 0:
                uni_to_oct_count += 1
            if note.vi == 0 and melody_info[i + 1].vi == 7:
                uni_to_oct_count += 1

        vscores.append(float(uni_to_oct_count) / len(melody_info))

        if debug:
            print "VSCORE 9: UNISON-OCTAVE LEAPS - ", uni_to_oct_count

        # vscore 10: The ending is unison or octave (ensured by generation)
        vscores.append(0.0)

        # vscore 11: all perfect intervals (also perfect fourth) should be approached by contrary or oblique motion
        bad_perfect_intervals_count = 0
        battuda = 0
        for i in range(1, len(melody_info)):
            if melody_info[i].beat == 0 and melody_info[i].vi in PERFECT_INTERVALS:
                bad_perfect_intervals_count += 1
                prev_cf = cantus_firmus[melody_info[i - 1].measure]
                prev_cp = melody_info[i - 1].pitch
                cur_cf = cantus_firmus[melody_info[i].measure]
                cur_cp = melody_info[i].pitch
                if prev_cp == cur_cp and prev_cf != cur_cf:  # oblique
                    bad_perfect_intervals_count -= 1
                if prev_cp > cur_cp and prev_cf <= cur_cf:  # contrary
                    bad_perfect_intervals_count -= 1
                    if melody_info[i].vi in [4, 7]:
                        battuda += 1
                if prev_cp < cur_cp and prev_cf >= cur_cf:  # contrary
                    bad_perfect_intervals_count -= 1
                    if melody_info[i].vi in [4, 7]:
                        battuda += 1

        vscores.append(float(bad_perfect_intervals_count) / len(cantus_firmus))

        if debug:
            print "VSCORE 11: PERF INTERVALS APPROACHED BADLY - ", bad_perfect_intervals_count

        # vscore 12: avoid simultaneous leaps in cf and cp, especially large leaps in same direction
        leaps_count = 0
        large_leaps_count = 0
        for i in range(len(melody_info) - 1):
            if melody_info[i + 1].beat == 0:
                leap_cp = melody_info[i + 1].pitch - melody_info[i].pitch
                leap_cf = cantus_firmus[melody_info[i + 1].measure] - cantus_firmus[melody_info[i].measure]
                if abs(leap_cf) > 1 and abs(leap_cp) > 1:
                    leaps_count += 1
                    if leap_cf > 6 and leap_cp > 6:
                        large_leaps_count += 1
                    if leap_cf < 6 and leap_cp < 6:
                        large_leaps_count += 1
        vscores.append(float(leaps_count + large_leaps_count) / (len(cantus_firmus) * 2))

        if debug:
            print "VSCORE 12: SIM LEAPS - ", leaps_count, ", LARGE SIM LEAPS - ", large_leaps_count

        # vscore 13: use all types of motion
        similar = 0
        contrary = 0
        oblique = 0
        parallel = 0
        for i in range(1, len(melody_info)):
            if melody_info[i].beat == 0:
                prev_cf = cantus_firmus[melody_info[i - 1].measure]
                prev_cp = melody_info[i - 1].pitch
                cur_cf = cantus_firmus[melody_info[i].measure]
                cur_cp = melody_info[i].pitch
                if prev_cp == cur_cp:
                    if prev_cf != cur_cf:
                        oblique = 1
                    else:
                        similar = 1
                if prev_cp > cur_cp:
                    if prev_cf <= cur_cf:
                        contrary = 1
                    else:
                        parallel = 1
                if prev_cp < cur_cp:
                    if prev_cf >= cur_cf:
                        contrary = 1
                    else:
                        parallel = 1
        types_of_motion = similar + oblique + contrary + parallel
        vscores.append(1 - float(types_of_motion) / 4)
        if debug:
            print "VSCORE 13: MOTION TYPES (SOCP) - ", similar, oblique, contrary, parallel

        # vscore 14: climax of the CF and CP should not coincide
        cf_highest_note = max(cantus_firmus)
        coincide = 0
        for note in melody_info:
            if note.pitch == highest_note and cantus_firmus[note.measure] == cf_highest_note:
                coincide = 1

        vscores.append(coincide)
        if debug:
            print "VSCORE 14: COINCIDE - ", coincide

        # vscore 15: Successive unisons, octaves and fifths on first beats are only valid
        # when separated by three quarter notes.
        bad_intervals_count = 0

        for i in range(len(melody_info) - 1):
            if melody_info[i].beat == 0 and melody_info[i].measure != len(cantus_firmus) - 1:
                if melody_info[i].vi in [0, 4, 7]:
                    separated = True
                    j = 1
                    while melody_info[i + j].measure == melody_info[i].measure:
                        if melody_info[i + j].duration > 2:
                            separated = False
                        j += 1
                    if melody_info[i + j].vi in [0, 4, 7] and not separated:
                        bad_intervals_count += 1
        vscores.append(float(bad_intervals_count) / (len(cantus_firmus) - 1))
        if debug:
            print "VSCORE 15: BAD INTERVALS - ", bad_intervals_count

        # vscore 16: successive unisons, octaves and fifths not on first beats:
        # valid when separated by at least 2 notes, otherwise not.
        # Unless it is a consonant suspension of quarter note: ok for afterbeat fifths and octaves
        # separated only by a single quearter.
        # ***what a complex rule, we don't care about consonant suspension due to storing format >_<***
        bad_intervals_count = 0

        for i in range(len(melody_info) - 2):
            separated = True
            if melody_info[i].beat != 0:
                if melody_info[i].vi in [0, 4, 7]:
                    if melody_info[i + 1].vi in [0, 4, 7]:
                        separated = False
                    if separated:
                        if melody_info[i + 2].vi in [0, 4, 7]:
                            separated = False
            if not separated:
                bad_intervals_count += 1

        vscores.append(float(bad_intervals_count) / (len(melody_info) - len(cantus_firmus)))

        if debug:
            print "VSCORE 16: BAD INTERVALS - ", bad_intervals_count

        # vscore 17: no ottava or quinta battuda, whatever it means
        vscores.append(float(battuda) / len(cantus_firmus))

        if debug:
            print "VSCORE 17: BATTUDAS - ", battuda

        # vscore 18: best ending: dissonant suspension into the leading tone (ensured by generation)
        vscores.append(0)

        # vscore 19: Thirds, sixths and tenths should predominate.
        good_interval_count = len(filter(lambda x: x.vi in [2, 5, 9], melody_info))
        if good_interval_count * 2 > len(melody_info):
            vscores.append(0.0)
        else:
            vscores.append(1.0 - float(2 * good_interval_count) / len(melody_info))

        if debug:
            print "VSCORE 19: 3RDS 6THS 10THS - ", good_interval_count

        genome.fitness = sum([x * y for x, y in zip(hscores, HSCORE_WEIGHTS)]) + \
                         sum([x * y for x, y in zip(vscores, VSCORE_WEIGHTS)])
        if debug:
            print "HSCORES: ", hscores
            print "VSCORES: ", vscores
            print "FINAL SCORE: ", genome.fitness
            print "FINAL SCORE UNSCALED: ", sum(hscores) + sum(vscores)
        return genome.fitness

    return fitness_function

def make_halt_function(cantus_firmus):
    """
    Returns a halt function for the given cantus firmus.
    """

    def halt(population, generation_count):
        """
        Given a population of candidate solutions and generation count (the
        number of epochs the algorithm has run) will return a boolean to
        indicate if an acceptable solution has been found within the referenced
        population.
        """
        return generation_count > DEFAULT_MAX_GENERATION or population[0].fitness == 0

    return halt

class Genome(ga.Genome):
    def mutate(self, mutation_range, mutation_rate, context):
        current_beat = 4
        cf_note_idx = 0
        for locus in range(1, len(self.chromosome)):
            if mutation_rate >= random.random():
                cf_note = context[cf_note_idx]
                if locus == 1:
                    valid_range = [0, 5, 7]
                elif locus == len(self.chromosome) - 1:
                    valid_range = [0, 7]
                elif locus == len(self.chromosome) - 5:
                    valid_range = [interval for interval in DISSONANCES if (interval + cf_note) < 17 and
                                   context[-2] - context[-3] + interval in CONSONANCES]
                    if len(valid_range) == 0:
                        valid_range = [interval for interval in range(1, 12) if (interval + cf_note) < 17 and
                                       context[-2] - context[-3] + interval in CONSONANCES]
                elif current_beat != 0:
                    valid_range = [interval for interval in range(1, 12) if (interval + cf_note) < 17 and
                                   abs(self.chromosome[locus][0] - cf_note - interval) <= mutation_range]
                else:
                    valid_range = [interval for interval in CONSONANCES if (interval + cf_note) < 17 and
                                   abs(self.chromosome[locus][0] - cf_note - interval) <= mutation_range]
                mutation = random.choice(valid_range)
                cp_note = cf_note + mutation
                self.chromosome[locus][0] = cp_note
            current_beat += self.chromosome[locus][1]
            cf_note_idx += current_beat / 8
            current_beat %= 8
        # Resets fitness score
        self.fitness = None
