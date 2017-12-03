import pywrapfst as fst

def CreatePrefixModel(word_model, prefix_model):
    '''
    Create a model handling character modelings (C<space>)*P.

    word_model will be modified and hence is the output of this function.
    '''
    # read opengrm output to fst
    word_model = fst.Fst.read(word_model)
    prefix_model = fst.Fst.read(prefix_model)

    # Create the trivial fst with only one space transition.
    space = fst.Fst()
    space.set_input_symbols(word_model.input_symbols())
    space.set_output_symbols(word_model.output_symbols())
    space.add_state()
    space.set_start(0)
    space_id = space.input_symbols().find("#")
    space.add_arc(0, fst.Arc(space_id, space_id, None, 1))
    space.add_state()
    space.set_final(1)

    # Compose the final machine.
    word_model.concat(space)
    word_model.closure()
    word_model.concat(prefix_model)
    word_model.arcsort(st="olabel")

    # save in to one file
    word_model.write("ch_lm.fst")

if  __name__ =='__main__':
    w_model = "brown_wd_alphabet.n5.kn.fst"
    pre_model = "brown_prefix_alphabet.n5.kn.fst"

    CreatePrefixModel(w_model, pre_model)
