
import sys

def is_all_char(s, ch):
  """
  All characters in 's' are 'ch'.
  """

  for c in s:
    if c != ch:
      return 0

  return 1

def cleanup_args(arglist):
  newarglist = list()

  for arg in arglist:
    newarg = arg.strip()
    while "_" == newarg[-1]:
      newarg = newarg[:-1]
    newarglist.append(newarg)

  return newarglist

class FuncDoc:
  def __init__(self, funcname, C_args, argdict):
    self.funcname = funcname
    self.C_args = C_args
    self.argdict = argdict

def getdoc(fname):
  txt = open(fname).read()
  purpose_idx = txt.find("Purpose")
  args_idx = txt.find("Arguments", purpose_idx)
  func_idx = txt.rfind('extern "C"', 1, purpose_idx)
  funcname = txt[txt.find("\n", func_idx)+1 : txt.find("(", func_idx)]
  C_args_idx = txt.find("(", func_idx) + 1
  # I'm counting on the fact that there is no ')' in arg list
  C_args = txt[C_args_idx : txt.find(")", C_args_idx)].split(",")
  C_args = cleanup_args(C_args)

  #print txt.find("\n", func_idx)+1 , txt.find("(", func_idx)
  #print fname, funcname, C_args, txt[C_args_idx:1]#, len(txt), args_idx ,func_idx 

  argdict = dict()
  argname = "_____" # dummy argument name
  argdict[argname] = list()

  # go through each "argline" in the section with arguments
  for argline in txt[args_idx:txt.find("*/", args_idx)].split("\n"):
    argfields = argline.split()

    switcharg = 0
    for inout in ("(input)", "(output)", "(input/output)", "(workspace)", "(input/workspace)", "(workspace/output)"):
      if len(argfields) > 1 and inout == argfields[1]:
        argname = argfields[0]
        arginout = inout
        argdict[argname] = [arginout]
        switcharg = 1
        break

    if not switcharg:
      argdict[argname].append(argline)

  for key in argdict:
    l = argdict[key]
    idx = len(l) - 1
    fields = l[idx].split()
    if len(fields) == 1 and is_all_char(fields[0], "="):
      del l[idx]

  return FuncDoc(funcname, C_args, argdict)

  for key in argdict:
    print key, argdict[key]

def getlatexdoc(funcdoc):
  latexdoc = "\\textsf{magma\_int\_t} "
  latexdoc += "\\textsf{\\textbf{%s}}" % funcdoc.funcname.replace("_", "\\_")
  latexdoc += "(\\textsf{"
  latexdoc += ",".join(funcdoc.C_args).replace("_", "\\_")
  latexdoc += "}); \\\n"

  latexdoc += "Arguments:\n\\begin{description}"

  for arg in funcdoc.C_args:
    argname = ""
    larg = len(arg)
    for idx in range(larg):
      ch = arg[larg-idx-1]
      if ch.isalpha():
        argname = ch + argname
      else:
        break

    latexdoc += "\\item[" + argname + "] "

    karg = argname.upper()
    if 0 and not funcdoc.argdict.has_key(karg):
      karg = "D" + karg

    try:
      ldoc = funcdoc.argdict[karg]
      latexdoc += ldoc[0] + "\n" # input/output/workspace
      latexdoc += "\\begin{verbatim}\n"

      # remove extra indentation
      lidx = len(ldoc[1]) - len(ldoc[1].lstrip())

      for l in ldoc[1:]:
        latexdoc += l[lidx:] + "\n"
      latexdoc += "\end{verbatim}\n"

    except:
      #sys.stderr.write("MAGMA %s\n" % " ".join(list((funcdoc.funcname, karg, str(funcdoc.argdict.keys()), latexdoc, str(funcdoc.argdict)))))
      pass

    latexdoc += "\n"

  latexdoc += "\\end{description}"

  return latexdoc

def main(argv):
  for fname in argv[1:]:
    funcdoc = getdoc(fname)
    latexdoc = getlatexdoc(funcdoc)
    print latexdoc

  return 0

if "__main__" == __name__ :
  sys.exit(main(sys.argv))
