
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
  def __init__(self, funcname, C_args, argdict, purpose, details):
    self.funcname = funcname
    self.C_args = C_args
    self.argdict = argdict
    self.purpose = purpose
    self.details = details

def getdoc(fname):
  txt = open(fname).read()
  purpose_idx = txt.find("\n", txt.find("Purpose")+14)+1
  args_idx = txt.find("Arguments", purpose_idx)
  func_idx = txt.rfind('extern "C"', 1, purpose_idx)
  funcname = txt[txt.find("\n", func_idx)+1 : txt.find("(", func_idx)]
  C_args_idx = txt.find("(", func_idx) + 1
  # I'm counting on the fact that there is no ')' in arg list
  C_args = txt[C_args_idx : txt.find(")", C_args_idx)].split(",")
  C_args = cleanup_args(C_args)

  details_idx = txt.find("   Further Details", args_idx)

  eoc_idx = txt.find("*/", args_idx)
  argend_idx = eoc_idx
  if details_idx > 0:
    argend_idx = details_idx

  if details_idx > 0:
    details_idx = txt.find("\n", details_idx + 30) + 1

  #print txt.find("\n", func_idx)+1 , txt.find("(", func_idx)
  #print fname, funcname, C_args, txt[C_args_idx:1]#, len(txt), args_idx ,func_idx 

  purpose = txt[purpose_idx:args_idx]
  if details_idx > 0:
    details = txt[details_idx:eoc_idx]
  else:
    details = ""

  argdict = dict()
  argname = "_____" # dummy argument name
  argdict[argname] = list()

  # go through each "argline" in the section with arguments
  for argline in txt[args_idx:argend_idx].split("\n"):
    argfields = argline.split()

    switcharg = 0
    for inout in ("(input)", "(output)", "(input/output)", "(workspace)", "(input/workspace)", "(workspace/output)", "(input"):
      if len(argfields) > 1 and inout == argfields[1]:
        argname = argfields[0]
        if ")" == inout[-1]:
          arginout = inout
        else:
          arginout = argline[argline.find("(") : argline.find(")")+1]
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

  purpose = txt[purpose_idx:args_idx]

  return FuncDoc(funcname, C_args, argdict, purpose, details)

  for key in argdict:
    print key, argdict[key]

def getlatexdoc(funcdoc):
  latexdoc = "\\textsf{magma\_int\_t} "
  latexdoc += "\\textsf{\\textbf{%s}}" % funcdoc.funcname.replace("_", "\\_")
  latexdoc += "(\\textsf{"
  latexdoc += ", ".join(funcdoc.C_args).replace("_", "\\_")
  latexdoc += "}); \\\n"

  latexdoc += "Arguments:\\\n\\begin{description}"

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

      lidx = len(ldoc[1])
      # remove extra indentation
      for l in ldoc[1:]:
        if len(l) > 0:
          lidx = min(lidx, len(l) - len(l.lstrip()))

      for l in ldoc[1:]:
        if l.rstrip():
          latexdoc += l[lidx:] + "\n"
      latexdoc += "\end{verbatim}\n"

    except:
      #sys.stderr.write("MAGMA %s\n" % " ".join(list((funcdoc.funcname, karg, str(funcdoc.argdict.keys()), latexdoc, str(funcdoc.argdict)))))
      pass

    latexdoc += "\n"

  latexdoc += "\\end{description}"

  return latexdoc

class HtmlDoc:
  def tag(self, tag, s):
    return "<" + tag + ">" + s + "</" + tag.split()[0] + ">\n"

  def begin_decl(self):
    return "<p>\n"

  def end_decl(self):
    return "</p>\n"

  def ret_type(self, s):
    return s + " "
    return self.tag('div class="ret_type"', s)

  def funcname(self, s):
    return self.tag('b', s)
    return self.tag('div class="funcname"', s)

  def decl_arg(self, s):
    return s
    return self.tag('div class="decl_arg"', s)

  def begin_purpose(self, s):
    return "<p><b>%s</b><br/>\n<pre>\n" % s

  def end_purpose(self, s):
    return "</pre>\n</p>\n"

  def begin_arguments(self, s):
    return "<p><b>%s</b><br/>\n" % s
    return self.tag('div class="arguments"', s) + "<ul>\n"

  def end_arguments(self, s):
    return "</ul>\n"

  def begin_arg(self, argname):
    return "<li><b>" + argname + "</b>\n"

  def end_arg(self, argname):
    return ""

  def inoutwork(self, s):
    return s + "\n"

  def begin_arg_desc(self, argname):
    return "<pre>\n"

  def arg_desc_line(self, s):
    return s + "\n"

  def end_arg_desc(self, argname):
    return "</pre>\n</li>\n"

  def begin_details(self, s):
    return "<p><b>%s</b><br/>\n<pre>\n" % s

  def end_details(self, s):
    return "</pre>\n</p>\n"

def getoutputdoc(fdoc, output):
  doc = output.begin_decl()
  doc += output.ret_type("magma_int_t")
  doc += output.funcname(fdoc.funcname)
  doc += "("
  doc += ", ".join(map(lambda s: output.decl_arg(s), fdoc.C_args))
  doc += ")\n"
  doc += output.end_decl()

  doc += output.begin_purpose("Purpose:")
  doc += fdoc.purpose
  doc += output.end_purpose("Purpose:")

  doc += output.begin_arguments("Arguments:")

  for arg in fdoc.C_args:
    argname = ""
    larg = len(arg)
    for idx in range(larg):
      ch = arg[larg-idx-1]
      if ch.isalpha():
        argname = ch + argname
      else:
        break

    doc += output.begin_arg(argname)
    doc += output.end_arg(argname)

    karg = argname.upper()
    if 0 and not funcdoc.argdict.has_key(karg):
      karg = "D" + karg

    ldoc = fdoc.argdict.get(karg, ["(missing)", "MISSING"])
    #sys.stderr.write("MAGMA %s\n" % " ".join(list((funcdoc.funcname, karg, str(funcdoc.argdict.keys()), latexdoc, str(funcdoc.argdict)))))

    if len(ldoc) > 1:
      lidx = len(ldoc[1])
    else:
      lidx = 0
    # remove extra indentation
    for l in ldoc[1:]:
      if len(l) > 0:
        lidx = min(lidx, len(l) - len(l.lstrip()))

    doc += output.inoutwork(ldoc[0]) # input/output/workspace
    doc += output.begin_arg_desc(argname)

    for l in ldoc[1:]:
      if l.rstrip():
        doc += output.arg_desc_line(l[lidx:])

    doc += output.end_arg_desc(argname)


  doc += output.end_arguments("Arguments:")

  if fdoc.details:
    doc += output.begin_details("Further Details:")
    doc += fdoc.details
    doc += output.end_details("Further Details:")

  return doc

def main(argv):
  for fname in argv[1:]:
    funcdoc = getdoc(fname)
    #latexdoc = getlatexdoc(funcdoc)
    doc = getoutputdoc(funcdoc, HtmlDoc())
    print doc

  return 0

if "__main__" == __name__ :
  sys.exit(main(sys.argv))
