%%
%% This is file `cqutthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% cqutthesis.dtx  (with options: `class')
%% 
%% This is a generated file.
%% 
%% Copyright (C) 2019-2022 by Miracle0565
%% Copyright (C) 2022-2023 by The-CCSN
%% 
%%     https://github.com/Miracle0565/CQUTthesis
%% 
%% This file may be distributed and/or modified under
%% the conditions of the LaTeX Project Public License,
%% either version 1.3c of this license or (at your option)
%% any later version. The latest version of this license
%% is in:
%% 
%%     http://www.latex-project.org/lppl.txt
%% 
%% and version 1.3c or later is part of all distributions
%% of LaTeX version 1999/12/01 or later.
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}[1999/12/01]
\ProvidesClass{cqutthesis}[2023/05/20 v2.0 CQUT Thesis Template]
\RequirePackage{ifxetex}
\RequireXeTeX
\RequirePackage{kvoptions}
\SetupKeyvalOptions{
    family   = cqut,
    prefix   = cqut@,
    setkeys  = \kvsetkeys,
}
\newif\ifcqut@bachelor\cqut@bachelorfalse
\newif\ifcqut@master\cqut@masterfalse
\newif\ifcqut@doctor\cqut@doctorfalse
\define@key{cqut}{type}{\expandafter\csname cqut@#1true\endcsname}
\DeclareBoolOption[false]{submit}
\DeclareBoolOption[false]{openany}
\DeclareComplementaryOption{openright}{openany}
\DeclareBoolOption[false]{debugger}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{ctexbook}}
\ProcessKeyvalOptions*
\PassOptionsToPackage{quiet}{xeCJK}
\ifcqut@openany\LoadClass[zihao=-4,UTF8,oneside]{ctexbook}
\else\LoadClass[zihao=-4,UTF8,openright]{ctexbook}\fi
\ifcqut@bachelor\relax\else
    \ifcqut@master\relax\else
        \ifcqut@doctor\relax\else
            \ClassError{cqutthesis}{%
                Specified thesis type is obligatory: \MessageBreak%
                type = [ bachelor | master | doctor ]%
            }{}
\fi\fi\fi
\NewDocumentCommand{\cqut@def@key}{ m o }{
    \@namedef{#1}##1{\@namedef{cqut@#1}{##1}}
    \define@key{cqut}{#1}[#2]{\@nameuse{#1}{##1}}
}
\newcommand{\cqutsetup}[1]{\kvsetkeys{cqut}{#1}}
\cqut@def@key{ChineseTitle}
\cqut@def@key{EnglishTitle}
\cqut@def@key{ChineseTitleLineA}
\cqut@def@key{ChineseTitleLineB}
\cqut@def@key{ChineseTitle}
\cqut@def@key{author}
\cqut@def@key{studentid}
\cqut@def@key{supervisor}
\cqut@def@key{major}
\cqut@def@key{date}[\the\year-\two@digits{\month}-\two@digits{\day}]
\ifcqut@bachelor
    \cqut@def@key{school}
    \cqut@def@key{class}
    \cqut@def@key{msupervisor}
\fi
\newcommand{\cqut@def@sep@key}[2]{%
    \@namedef{#1}##1{%
        \@namedef{cqut@#1}{}%
        \@for\reserved@a:=##1\do{
            \expandafter\ifx\csname cqut@#1\endcsname\@empty\else
            \expandafter\g@addto@macro\csname cqut@#1\endcsname{#2}\fi
            \expandafter\expandafter\expandafter\g@addto@macro%
            \expandafter\csname cqut@#1\expandafter\endcsname%
            \expandafter{\reserved@a}
        }
    }
    \define@key{cqut}{#1}{\@nameuse{#1}{##1}}%
}
\cqut@def@sep@key{ChineseKeywords}{\cqut@ChineseKeywords@sep}
\cqut@def@sep@key{EnglishKeywords}{\cqut@EnglishKeywords@sep}
\RequirePackage{geometry,fancyhdr}
\RequirePackage{titletoc}
\RequirePackage{amsmath,amsthm,amssymb,unicode-math}
\RequirePackage{pifont,enumitem}
\RequirePackage{siunitx}
\RequirePackage[version = 4]{mhchem}
\RequirePackage{float}
\RequirePackage{longtable,threeparttable,tabularx,multirow,booktabs}
\RequirePackage{graphicx}
\RequirePackage[labelformat = simple]{subcaption}
\RequirePackage[format=hang]{caption}
\RequirePackage{bicaption}
\RequirePackage{tikz}
\RequirePackage{calc}
\RequirePackage[absolute,overlay]{textpos}
\RequirePackage{xeCJKfntef}
\RequirePackage{listings}
\RequirePackage[sort&compress]{gbt7714}
\RequirePackage{xcolor}
\RequirePackage{pdfpages}
\RequirePackage[bottom,perpage]{footmisc}
\RequirePackage{xpatch}
\RequirePackage{xifthen}
\ifcqut@debugger
    \RequirePackage[color=blue,type=upperleft,showframe,hshift=2.7cm,vshift=3.5cm]{fgruler}
    \RequirePackage[pagewise]{lineno}
\fi
\RequirePackage{hyperref}
\newcommand\cqut@fontsetval{\csname g__ctex_fontset_tl\endcsname}
\ifthenelse{\equal{\cqut@fontsetval}{windows}}{
    \xeCJKsetup{EmboldenFactor=2.2,SlantFactor=0.2}
    \setCJKmainfont{SimSun}[AutoFakeBold,AutoFakeSlant]
    \setCJKsansfont{SimHei}[AutoFakeBold]
}
{
    \ClassWarning{cqutthesis}{%
        The fonts differ to the default.\MessageBreak%
        Use fontset = windows to correct.%
    }
}
\newcommand{\bfsong}{\bfseries}
\newcommand{\bfhei}{\sffamily\bfseries}
\setmainfont{Times New Roman}
\setsansfont{Times New Roman}
\setmonofont{Times New Roman}
\IfFontExistsTF{LibertinusMath-Regular.otf}{%
    \setmathfont{latinmodern-math.otf}
    \setmathfont[%
        range = {%
            up/{latin,Latin,num},
            it/{latin,Latin,greek},
            bfit/{latin,Latin},
        }
    ]{LibertinusMath-Regular.otf}
}{}
\xeCJKDeclareCharClass{CJK}{"2460->"2469}
\newcommand{\cqut@headfont}{\zihao{-5}\songti}
\newcommand{\cqut@footfont}{\zihao{-5}\songti}
\ifcqut@bachelor
    \newcommand{\cqut@dcl@titlefont}{\zihao{3}\songti}
    \newcommand{\cqut@dcl@ideclarefont}{\zihao{4}\songti}
\else
    \newcommand{\cqut@dcl@titlefont}{\zihao{4}\bfseries}
    \newcommand{\cqut@dcl@textfont}{\zihao{4}\mdseries}
    \newcommand{\cqut@auth@titlefont}{\zihao{4}\bfseries}
    \newcommand{\cqut@auth@textfont}{\zihao{4}\mdseries}
\fi
\ifcqut@bachelor
    \newcommand{\cqut@abs@titfont}{\zihao{3}\bfsong}
    \newcommand{\cqut@abs@infofont}{\zihao{5}\mdseries}
    \newcommand{\cqut@abs@absfont}{\zihao{4}\songti}
    \newcommand{\cqut@keywordsfont}{\zihao{4}\heiti\CJKfamily+{}}
    \newcommand{\cqut@abs@titfonten}{\zihao{3}\bfseries}
    \newcommand{\cqut@abs@absfonten}{\zihao{4}\songti}
    \newcommand{\cqut@keywordsfonten}{\zihao{4}\heiti\CJKfamily+{}}
\else
    \newcommand{\cqut@abs@titfont}{\zihao{3}\heiti}
    \newcommand{\cqut@abs@absfont}{\zihao{-3}\heiti}
    \newcommand{\cqut@abs@textfont}{\zihao{4}\songti}
    \newcommand{\cqut@keywords@titfont}{\zihao{4}\heiti}
    \newcommand{\cqut@keywords@font}{\zihao{4}\songti}
    \newcommand{\cqut@abs@titfonten}{\zihao{3}\bfseries}
    \newcommand{\cqut@abs@absfonten}{\zihao{-3}\bfseries}
    \newcommand{\cqut@keywords@titfonten}{\zihao{4}\bfseries}
    \newcommand{\cqut@keywords@fonten}{\zihao{4}\mdseries}
\fi
\ifcqut@bachelor
    \newcommand{\cqut@toc@tocfont}{\zihao{4}\mdseries\heiti}
\else
    \newcommand{\cqut@toc@tocfont}{\zihao{3}\heiti}
\fi
\newcommand{\cqut@toc@chapfont}{\zihao{4}\heiti}
\newcommand{\cqut@toc@secfont}{\zihao{4}\songti}
\newcommand{\cqut@toc@ssecfont}{\zihao{4}\songti}
\newcommand{\cqut@toc@sssecfont}{\zihao{-4}\songti}
\newcommand{\cqut@toc@dsgfigfont}{\cqut@toc@ssecfont}
\newcommand{\cqut@toce@chapfont}{\zihao{4}\bfseries}
\ifcqut@bachelor
    \newcommand{\cqut@chapfont}{\zihao{-3}\bfsong\centering}
    \newcommand{\cqut@secfont}{\zihao{-3}\bfsong\centering}
    \newcommand{\cqut@ssecfont}{\zihao{-4}\bfsong\raggedright}
    \newcommand{\cqut@sssecfont}{\zihao{-4}\songti}
\else
    \newcommand{\cqut@chapfont}{\zihao{3}\heiti\centering}
    \newcommand{\cqut@secfont}{\zihao{4}\heiti\raggedright}
    \newcommand{\cqut@ssecfont}{\zihao{-4}\heiti\raggedright}
    \newcommand{\cqut@sssecfont}{\cqut@ssecfont}
\fi
\newcommand{\cqut@floatfont}{\zihao{5}}
\newcommand{\cqut@float@captionfont}{\zihao{5}\mdseries}
\newcommand{\cqut@float@labelfont}{\zihao{5}\mdseries}
\newcommand{\cqut@codefont}{\ttfamily\songti\zihao{5}}
\newcommand{\cqut@codecomfont}{%
    \ttfamily\slshape
    \ifcqut@submit\relax\else\color{gray!80!black}\fi}
\newcommand{\cqut@codekeyfont}{%
    \ttfamily\bfseries
    \ifcqut@submit\relax\else\color{blue!80!black}\fi}
\ifcqut@bachelor
    \newcommand\cqut@headtext{化工原理课程设计}
    \newcommand\cqut@foottext{\thepage}
    \newcommand\cqut@dcl@title{诚信声明}
    \newcommand\cqut@dcl@ideclare{本人声明：}
    \newcommand\cqut@dcl@text{
        所呈交的学位论文是本人在导师的指导下独立进行研究工作所取得的成果。
        据我所知，除文中已经注明引用的内容外，本论文不包含任何其他个人或集体
        已经发表或撰写过的研究成果，也不包含为获得重庆理工大学或其它教育机构
        的学位或证书而使用过的材料。对论文所涉及的研究工作做出贡献的其他个人
        和集体，均已在文中以明确方式标明或致谢。本人完全意识到本声明的法律
        结果由本人承担。
    }
    \newcommand\cqut@dcl@sign{本人签名：\hspace{14em}年\hspace{2.5em}月\hspace{2.5em}日}
    \newcommand\cqut@task@title{本科生课程设计（论文）任务书}
    \newcommand\cqut@task@pdf{任务书}
    \newcommand\cqut@task@ChineseTitle{设计（论文）题目：}
    \newcommand\cqut@task@school{学院：}
    \newcommand\cqut@task@major{专业：}
    \newcommand\cqut@task@class{班级：}
    \newcommand\cqut@task@student{学生：}
    \newcommand\cqut@task@supervisor{指导教师（含职称）：}
    \newcommand\cqut@task@msupervisor{专业负责人：}
    \newcommand\cqut@task@itema{1.设计（论文）的主要任务及目标}
    \newcommand\cqut@task@itemb{2.设计（论文）的基本要求和内容}
    \newcommand\cqut@task@itemc{3.主要参考文献}
    \newcommand\cqut@task@itemd{4.进度安排}
    \newcommand\cqut@cabstract@title{摘要}
    \newcommand\cqut@eabstract@title{ABSTRACT}
    \newcommand\cqut@ChineseKeywords@title{关键词：}
    \newcommand\cqut@EnglishKeywords@title{Keywords:~}
    \newcommand\cqut@ChineseKeywords@sep{\quad}
    \newcommand\cqut@EnglishKeywords@sep{; }
    \newcommand\cqut@supervisor@title{指导教师：}
    \newcommand\cqut@content@title{目\hspace{\ccwd}录}
    \newcommand\cqut@dfigurename{设计图纸}
    \newcommand\cqut@dfigure@content@title{设计图纸目录}
    \newcommand\cqut@foreword@title{前言}
    \newcommand\cqut@concl@title{结论}
    \newcommand\cqut@trans@title{翻译}
    \newcommand\cqut@ack@title{致谢}
    \newcommand\cqut@deno@title{符号说明}
    \renewcommand\lstlistingname{代码}
\else
    \newcommand\cqut@headtext{\leftmark}
    \newcommand\cqut@foottext{\thepage}
    \newcommand\cqut@dcl@title{重庆理工大学学位论文原创性声明}
    \newcommand\cqut@dcl@text{
        本人郑重声明：所呈交的学位论文，是本人在导师的指导下，独立
        进行研究工作所取得的成果。除文中已经注明引用的内容外，本论
        文不含任何其他个人或集体已经发表或撰写过的作品成果。对本文
        的研究做出重要贡献的个人和集体，均已在文中以明确方式标明。
        本人完全意识到本声明的法律结果由本人承担。
    }
    \newcommand\cqut@auth@title{关于论文使用授权的说明}
    \newcommand\cqut@auth@text{
        学位论文作者完全了解重庆理工大学有关保留和使用学位论文的规定，
        即：研究生在校攻读学位期间论文工作的知识产权单位属重庆理工大学。
        学校有权保留并向国家有关部门或机构送交论文的复印件和磁盘，允许
        学位论文被查阅和借阅；学校可以公布学位论文的全部或部分内容，
        可以允许采用影印、缩印或其它复制手段保存、汇编学位论文。

        $\Box$ 论文暂不公开（或保密）注释：本学位论文属于暂不公开
        （或保密）范围，在\rule{1.5em}{.4pt}年解密后适用本授权书。

        $\Box$ 非暂不公开（或保密）论文注释：本学位论文不属于暂不公开
        （或保密）范围，适用本授权书。
    }
    \newcommand\cqut@dcl@pdf{版权页}
    \newcommand\cqut@sign@author{作者签名：\rule{7em}{.4pt}\qquad{日期：}\rule{7em}{.4pt}}
    \newcommand\cqut@sign@supervisor{导师签名：\rule{7em}{.4pt}\qquad{日期：}\rule{7em}{.4pt}}
    \newcommand\cqut@cabstract@title{摘\hspace{\ccwd}要}
    \newcommand\cqut@eabstract@title{ABSTRACT}
    \newcommand\cqut@ChineseKeywords@title{关键词：}
    \newcommand\cqut@EnglishKeywords@title{KEY WORDS:~}
    \newcommand\cqut@ChineseKeywords@sep{，}
    \newcommand\cqut@EnglishKeywords@sep{,~}
    \newcommand\cqut@content@title{目\hspace{\ccwd}录}
    \newcommand\cqut@econtent@chapter{Chapter~}
    \newcommand\cqut@econtentsname{Contents}
    \newcommand\cqut@deno@title{符号和缩略词说明}
    \newcommand\cqut@ack@title{致谢}
    \newcommand\cqut@eack@title{Acknowledgement}
    \newcommand\cqut@achi@title{研究成果及发表的学术论文}
    \newcommand\cqut@eachi@title{Research Results and Published Academic Papaers}
    \newcommand\cqut@resu@title{作者及导师简介}
    \newcommand\cqut@eresu@title{About the Author}
    \newcommand\equationname{式}
    \newcommand\cqut@bitable@name{Table}
    \newcommand\cqut@bifigure@name{Figure}
    \newcommand\cqut@ebib@title{Bibliography}
    \renewcommand\lstlistingname{代码}
\fi
\geometry{%
    a4paper,nomarginpar,
    top = 3.5cm, bottom = 2.6cm, left = 2.7cm, right = 2.7cm,
    headheight = 1.1cm, footskip = 0.6cm,
}
\fancypagestyle{plain}{%
    \fancyhf{}
    \fancyhead[C]{\cqut@headfont\cqut@headtext}
    \fancyfoot[C]{\cqut@footfont\cqut@foottext}
    \renewcommand\headrulewidth{0.6pt}
    \renewcommand\footrulewidth{0pt}
}
\pagestyle{plain}
\xpatchcmd{\frontmatter}
    {\pagenumbering{roman}}
    {\pagenumbering{Roman}}
    {}{}
\xpatchcmd{\cleardoublepage}
    {\newpage}
    {\thispagestyle{empty}\newpage}
    {}{}
%% \subsection{封面}
\newcommand\cqut@title@ul[1]{%
    \CJKunderline{\makebox[\widthof{\cqut@ChineseTitleLineA\enspace}][l]{#1}}
}
\newcommand\cqut@major@ul[1]{%
    \CJKunderline{\makebox[\widthof{\cqut@major\quad}][c]{#1}}
}
\newcommand\cqut@format@date[1]{%
    \edef\cqut@@date{\cqut@date}%
    \def\cqut@@process@date##1-##2-##3\@nil{%
        #1{##1}{##2}{##3}%
    }%
    \expandafter\cqut@@process@date\cqut@@date\@nil
}
\newcommand\@zhdate[3]{%
    \bfseries\zhkai
    \ifcqut@bachelor
        \zihao{4}
        #1年\makebox[1.5em]{#2}月\makebox[1.5em]{#3}日
    \else
        \Large\ziju{.1}%
        日\hspace{1.5\ccwd}期：\zhdigits{#1}年\zhnumber{#2}月\zhnumber{#3}日%
    \fi
}
\newcommand{\cqut@makecover}{%
    \newgeometry{
        top = 2.1cm, bottom = 2.2cm,
        left = 2cm, right = 2cm,
        headheight = 0cm, footskip = 0cm
    }
    \thispagestyle{empty}
    \xeCJKsetup{EmboldenFactor=2.3}
    \begin{table}[h]
        \raggedleft
        \ifcqut@bachelor
            \songti\zihao{5}
        \else
            \renewcommand{\arraystretch}{1.3}\large
        \fi
        \begin{tabular}{l@{}l}
            \ifcqut@bachelor%
                班\hspace{2\ccwd}级&：\CJKfamily+{zhsong}\cqut@class\hspace*{2\ccwd}\\
                学\hspace{2\ccwd}号&：\CJKfamily+{zhsong}\cqut@studentid\hspace*{2\ccwd}\\
            \else%
                \xbsong 单位代码 & ：\CJKfamily+{zhsong}\mbox{10010}\\
                \xbsong 学\hspace{2\ccwd}号 & ：\CJKfamily+{zhsong}\mbox{\cqut@studentid}\\
            \fi
        \end{tabular}
    \end{table}
    \ifcqut@bachelor\vspace{3bp}\else\vspace{26bp}\fi
    \begin{minipage}[c][][c]{2.35cm}
        \includegraphics[width = 2.35cm]{figure/CQUT-badge.pdf}
    \end{minipage}
    \qquad
    \begin{minipage}[c][2.35cm][c]{9cm}
        \xingkai
        \ifcqut@bachelor
            \ziju{0.1}\hspace*{0.8\ccwd}\zihao{-0}
        \else
            \ziju{-0.02}\zihao{0}
        \fi
        {重庆理工大学}
    \end{minipage}
    \ifcqut@bachelor\vspace{12bp}\else\vspace{18bp}\fi
    \begin{center}
        \ifcqut@bachelor
            \ziju{0.15}\xbsong\zihao{-0}\CJKfamily+{}{%
            \punctstyle{banjiao}课程设计（论文）% (论文)
            }
        \else%
            %\ziju{-0.05}
            \dbsong\fontsize{38bp}{60bp}\selectfont
            \ifcqut@master{硕}\else{博}\fi{}士研究生学位论文
        \fi
    \end{center}
    \ifcqut@bachelor
            \setlength{\TPHorizModule}{1cm}
            \setlength{\TPVertModule}{1cm}
            \begin{textblock}{0}(0,10)
                \noindent%
                \tikz[line width=1.2pt] \draw (0,0) -- (21,0) (0,0.2) -- (21,0.2);
            \end{textblock}
            \vspace{72bp}
            \begin{center}
                \xeCJKsetup{
                    underline = {
                        skip      = false,
                        thickness = 1.4pt,
                    }
                }
                \renewcommand{\arraystretch}{2}
                \begin{tabular}{%
                        >{\zihao{-2}\xbsong}l
                        >{\zihao{3}\bfseries}l
                }
                    题\hspace{\ccwd}目  & \cqut@title@ul{\cqut@ChineseTitleLineA}\\
                    \ifthenelse{\equal{\cqut@ChineseTitleLineB}{}}{\relax}{
                                        & \cqut@title@ul{\cqut@ChineseTitleLineB}\\
                    }
                \end{tabular}
                \vskip144bp\zihao{3}
                \begin{tabular}{
                        >{\begin{CJKfilltwosides}[b]{4\ccwd}\xbsong}r%
                            <{\end{CJKfilltwosides}}%
                        @{}>{\enspace\zihao{4}\bfseries}l%
                        <{\vspace{.2em}}%
                    }
                    专业     & \cqut@major@ul{\cqut@major}\\
                    学生     & \cqut@major@ul{\cqut@author}\\
                    指导教师 & \cqut@major@ul{\cqut@supervisor}\\
                \end{tabular}
            \end{center}
            \vfill{%
                \begin{center}%
                    \cqut@format@date{\@zhdate}%
                \end{center}%
            }
    \else
        \vspace{76bp}
        \begin{center}
            \xeCJKsetup{
                underline = {
                    skip      = false,
                    thickness = 1.2pt,
                }
            }
            \renewcommand{\arraystretch}{2.7}
            \begin{tabular}{
                >{\fontsize{20}{36}\selectfont\begin{CJKfilltwosides}[b]{4\ccwd}\sffamily\bfseries}r%
                <{\end{CJKfilltwosides}}%
                @{}>{\enspace\zihao{3}\bfseries}l%
                <{\vspace{.2em}}%
            }
                题目     & \cqut@title@ul{\cqut@ChineseTitleLineA}\\
                \ifthenelse{\equal{\cqut@ChineseTitleLineB}{}}{\relax}{
                                        & \cqut@title@ul{\cqut@ChineseTitleLineB}\\
                }
                专业     & \cqut@title@ul{\zihao{4}\cqut@major}\\
                研究生   & \cqut@title@ul{\zihao{4}\cqut@author}\\
                指导教师 & \cqut@title@ul{\zihao{4}\cqut@supervisor}\\
            \end{tabular}
        \end{center}
        \vfill{%
            \begin{center}%
                \cqut@format@date{\@zhdate}%
            \end{center}%
        }
    \fi
}
\NewDocumentCommand{\makecover}{ o }{%
    \hypertarget{cover}{}
    \pdfbookmark{封面}{cover}
    \IfNoValueTF{#1}{
        \IfFontExistsTF{FZXiaoBiaoSong-B05S}{%
            \newCJKfontfamily[xbsong] \xbsong {FZXiaoBiaoSong-B05S}
            \newCJKfontfamily[dbsong] \dbsong {FZDaBiaoSong-B06S}
            \newCJKfontfamily[xingkai]\xingkai{STXingkai}
            \newCJKfontfamily[zhkai]  \zhkai  {KaiTi} [AutoFakeBold]
            \cqut@makecover
        }{
            \cleardoublepage
        }

    }{%
        \includepdf[pages=-]{#1}
    }
\restoregeometry
}
\newcommand{\cqut@pdfmark}[2]{%
    \hypertarget{#2}{}
    \pdfbookmark{#1}{#2}
}
\newcommand{\cqut@makedeclare}{
    \ifcqut@bachelor
        \centerline{\cqut@dcl@titlefont\cqut@dcl@title}\par\vspace{1em}
        \leftline{\cqut@dcl@ideclarefont\cqut@dcl@ideclare}%
        \cqut@dcl@text
        \par\vspace{2em}\hfill{\cqut@dcl@sign}
    \else{
        \thispagestyle{empty}
        \linespread{1.5534}
        \setlength{\parskip}{15pt}
        \centerline{\cqut@dcl@titlefont\cqut@dcl@title}
        \par\indent\cqut@dcl@textfont\cqut@dcl@text
        \par\indent\cqut@sign@author\\[30pt]
        \centerline{\cqut@auth@titlefont\cqut@auth@title}
        \par\indent\cqut@auth@textfont\cqut@auth@text
        \par\indent\cqut@sign@author
        \par\indent\cqut@sign@supervisor
    }\fi
    \clearpage\normalsize
}
\NewDocumentCommand{\makedeclare}{ o }{%
    \cqut@pdfmark{\ifcqut@bachelor\cqut@dcl@title\else\cqut@dcl@pdf\fi}{declare}
    \IfNoValueTF{#1}{\cqut@makedeclare}{%
        \includepdf[pages=-]{#1}
    }
}
\ifcqut@bachelor
    \NewDocumentEnvironment{taskbook}{ s }{%
        \cleardoublepage%
        \cqut@pdfmark{\cqut@task@pdf}{taskbook}%
        \section*{\cqut@task@title}%
        \IfBooleanTF{#1}{\pagestyle{empty}}{}%
    }{\clearpage}
    \NewDocumentCommand{\cqut@ul}{ o m }{\underline{\makebox[#1][c]{#2}}}
    \NewDocumentCommand\taskinfo{ s }{%
        \noindent%
        \begin{minipage}{\textwidth}%
            \punctstyle{plain}\linespread{1.94}%
                \cqut@task@ChineseTitle\cqut@ul[27em]{\cqut@ChineseTitle}\\[3pt]
            \IfBooleanTF{#1}{% less lines break
                \cqut@task@school\cqut@ul[10em]{\cqut@school}\quad
                \cqut@task@major\cqut@ul[10em]{\cqut@major}\quad
                \cqut@task@class\cqut@ul[5em]{\cqut@class}\\[3pt]
                \cqut@task@student\cqut@ul[5em]{\cqut@author}\quad
                \cqut@task@supervisor\cqut@ul[6em]{\cqut@supervisor}\quad
                \cqut@task@msupervisor\cqut@ul[4em]{\cqut@msupervisor}%
            }{% more lines break
                \cqut@task@school\cqut@ul[15em]{\cqut@school}\qquad
                \cqut@task@major\cqut@ul[13em]{\cqut@major}\\[3pt]
                \cqut@task@class\cqut@ul[15em]{\cqut@class}\qquad
                \cqut@task@student\cqut@ul[13em]{\cqut@author}\\[3pt]
                \cqut@task@supervisor\cqut@ul[8em]{\cqut@supervisor}\qquad
                \cqut@task@msupervisor\cqut@ul[10em]{\cqut@msupervisor}%
            }
        \end{minipage}\vspace{1.5em}
    }
    \newcounter{taskitemcnt}
    \newcommand{\cqut@taskitem}{%
        \stepcounter{taskitemcnt}%
        \ifcase\value{taskitemcnt}\relax\or%
        {\cqut@task@itema}\or%
        {\cqut@task@itemb}\or%
        {\cqut@task@itemc}\or%
        {\cqut@task@itemd}\else%
        \ClassError{cqutthesis}{Too many \protect\taskitem\space used}%
        {Keep \protect\taskitem\space no more than 4 times.}\fi
    }
    \newcommand{\taskitem}{\vspace{6pt}\noindent\cqut@taskitem\par}
\fi
\newenvironment{bibenumerate}{%
    \begin{enumerate}[label={[\arabic*]},leftmargin=3em]
}{\end{enumerate}}
\ifcqut@bachelor
    \newenvironment{cabstract}{%
        \cleardoublepage
        \cqut@pdfmark{\cqut@cabstract@title}{cabstract}
        \begin{center}
            \cqut@abs@titfont\cqut@ChineseTitle
            \vskip22pt
            \cqut@abs@infofont\cqut@author\quad%
            \cqut@class\quad\cqut@studentid\quad%
            \cqut@supervisor@title\cqut@supervisor%
            \vskip22pt
            \cqut@abs@absfont\cqut@cabstract@title
        \end{center}\par\vspace{18pt}
    }{\vskip1em\noindent\cqut@keywordsfont\cqut@ChineseKeywords@title\cqut@ChineseKeywords}
\else
    \newenvironment{cabstract}{%
        \cleardoublepage
        \cqut@pdfmark{\cqut@cabstract@title}{cabstract}\markboth{\cqut@cabstract@title}{\cqut@cabstract@title}
        \begin{center}
            \cqut@abs@titfont\cqut@ChineseTitle
            \vskip24pt
            \cqut@abs@absfont\cqut@cabstract@title
        \end{center}\par\vspace{12pt}
        \cqut@abs@textfont
    }{\vskip24pt%
    \noindent\cqut@keywords@titfont\cqut@ChineseKeywords@title\cqut@keywords@font\cqut@ChineseKeywords}
\fi
\ifcqut@bachelor
    \newenvironment{eabstract}{%
        \cleardoublepage%
        \cqut@pdfmark{\cqut@eabstract@title}{abstracten}
        \begin{center}
            \cqut@abs@titfonten\cqut@EnglishTitle\vskip22pt
            \cqut@abs@absfonten\cqut@eabstract@title
        \end{center}
        \par\vspace{22pt}
    }{\vskip1em\noindent\cqut@keywordsfonten \cqut@EnglishKeywords@title\cqut@EnglishKeywords}
\else
    \newenvironment{eabstract}{%
        \cleardoublepage%
        \cqut@pdfmark{\cqut@eabstract@title}{eabstract}\markboth{\cqut@eabstract@title}{\cqut@eabstract@title}
            \begin{center}
                \setlength{\leftskip}{3em plus 1em minus .5em}
                \setlength{\rightskip}{3em plus 1em minus .5em}
                \cqut@abs@titfonten\cqut@EnglishTitle\vskip76pt
                \cqut@abs@absfonten\cqut@eabstract@title
            \end{center}\par\vspace{40pt}
        \cqut@abs@textfont
    }{\vskip48pt%
    \noindent\cqut@keywords@titfonten\cqut@EnglishKeywords@title\cqut@keywords@fonten\cqut@EnglishKeywords}
\fi
\setcounter{secnumdepth}{3}
\ifcqut@bachelor
    \setcounter{tocdepth}{2}
\else
    \setcounter{tocdepth}{3}
\fi
\renewcommand\tableofcontents{%
    \cleardoublepage
    \pdfbookmark{\contentsname}{toc}%
    \chapter*{\cqut@toc@tocfont\cqut@content@title}%
    \ifcqut@bachelor\relax\else\markboth{\cqut@content@title}{\cqut@content@title}\fi
    \@starttoc{toc}
}
\ifcqut@bachelor\relax\else
    \newcommand\tableofcontentsEN{%
        \cleardoublepage
        \pdfbookmark{\cqut@econtentsname}{toce}%
        \chapter*{%
            \em\cqut@econtentsname%
            \ifcqut@bachelor\relax\else\markboth{\cqut@econtentsname}{\cqut@econtentsname}\fi%
        }%
        \@starttoc{toce}
    }
    \newcommand\cqut@addengcontents[3]{%
        \addcontentsline{toce}{#1}{\protect\numberline{#3\@nameuse{the#1}}#2}
    }
    \newcommand\echapter[1]{\cqut@addengcontents{chapter}{#1}{\cqut@toce@chapfont{}\cqut@econtent@chapter}}
    \newcommand\esection[1]{\cqut@addengcontents{section}{#1}{}}
    \newcommand\esubsection[1]{\cqut@addengcontents{subsection}{#1}{}}
    \newcommand\esubsubsection[1]{\cqut@addengcontents{subsubsection}{#1}{}}
    \NewDocumentCommand{\bichapter}{ o m m }{\chapter[#1]{#2}\echapter{#3}}
    \NewDocumentCommand{\bisection}{ o m m }{\section[#1]{#2}\esection{#3}}
    \NewDocumentCommand{\bisubsection}{ o m m }{\subsection[#1]{#2}\esubsection{#3}}
    \NewDocumentCommand{\bisubsubsection}{ o m m }{\subsubsection[#1]{#2}\esubsubsection{#3}}
\fi
\ifcqut@bachelor
    \titlecontents{chapter}[0em]
        {\cqut@toc@chapfont\vspace{3pt}}
        {\thecontentslabel\quad}{}
        {\titlerule*[.4em]{$\cdot$}\bfseries\contentspage}
    \titlecontents{section}[2.5em]
        {\cqut@toc@secfont\vspace{-3pt}}
        {\thecontentslabel\quad}{}
        {\titlerule*[.5em]{$\cdot$}\contentspage}
    \titlecontents{subsection}[4.0em]
        {\cqut@toc@ssecfont\vspace{-3pt}}
        {\thecontentslabel\hspace{0.70cm}}{}
        {\titlerule*[1em]{$\cdot$}\contentspage}
\else
    \titlecontents{chapter}[0em]
        {\cqut@toc@chapfont\vspace{3pt}}
        {\thecontentslabel\quad}{}
        {\titlerule*[.25em]{$\cdot$}\bfseries\contentspage}
    \titlecontents{section}[2.25em]
        {\cqut@toc@secfont\vspace{-3pt}}
        {\thecontentslabel\quad}{}
        {\titlerule*[.5em]{$\cdot$}\contentspage}
    \titlecontents{subsection}[5em]
        {\cqut@toc@ssecfont\vspace{-3pt}}
        {\thecontentslabel\quad}{}
        {\titlerule*[1em]{$\cdot$}\contentspage}
    \titlecontents{subsubsection}[8.5em]
        {\cqut@toc@sssecfont\vspace{-3pt}}
        {\thecontentslabel\quad}{}
        {\titlerule*[1em]{$\cdot$}\contentspage}
\fi
\renewcommand{\bibsection}{
    \chapter*{\bibname}
    \addcontentsline{toc}{chapter}{\bibname}
    \ifcqut@bachelor\relax\else
        \addcontentsline{toce}{chapter}{\cqut@toce@chapfont\cqut@ebib@title}
        \markboth{\bibname}{\bibname}
    \fi
}
\ifcqut@bachelor
	\newcommand{\dcaption}[1]{%
		\refstepcounter{figure} % 确保图纸计数器递增
		\caption{#1}
		\addcontentsline{toc}{section}{%
			\hspace{2.5em}\cqut@dfigurename~\thefigure\hspace{1em}{#1}%
		}
	}
    \newcommand{\cqut@listof}[2]{%
        \xpatchcmd{\@dottedtocline}{\hbox{.}}{\hbox{$\cdot$}}{}{}
        \renewcommand{\@dotsep}{1.7}%
        \@ifundefined{ext@#1}{\float@error{#1}}{%
            \@namedef{l@#1}{\@dottedtocline{1}{0em}{2em}}
            \float@listhead{\cqut@toc@tocfont{}#2}%
            \begingroup\setlength{\parskip}{\z@}%
            \cqut@toc@dsgfigfont\@starttoc{\@nameuse{ext@#1}}%
            \endgroup%
        }
    }
    \newcommand{\listofdesignfigures}{%
        \cleardoublepage%
        \cqut@pdfmark{\cqut@dfigure@content@title}{dfigure}%
        \cqut@listof{dfigure}{\cqut@dfigure@content@title}
    }
\fi
\newenvironment{foreword}{%
    \cleardoublepage
    \cqut@pdfmark{\cqut@foreword@title}{foreword}
    \chapter*{\cqut@foreword@title}
}{}
\ifcqut@bachelor
    \linespread{1.5221}
\else
    \linespread{1.3837}
\fi
\ifcqut@bachelor
    \ctexset{%
        chapter = {%
            name        = {第,章},
            number      = \arabic{chapter},
            format      = \cqut@chapfont,
            beforeskip  = 12 pt,
            afterskip   = 32 pt,
            fixskip     = true,
            aftername   = \qquad,
        },
        section = {%
            name        = {第,节},
            format      = \cqut@secfont,
            aftername   = \quad,
            beforeskip  = 29 pt,
            afterskip   = 32 pt,
            fixskip     = true,
        },
        subsection = {%
            format      = \cqut@ssecfont,
            aftername   = \quad,
            beforeskip  = 3 pt,
            afterskip   = 3 pt,
        },
        subsubsection = {%
            name        = {(,)},
            format      = \cqut@sssecfont,
            number      = \arabic{subsubsection},
            beforeskip  = 3 pt,
            afterskip   = 3 pt,
            aftername   = \hspace{0.5em},
            indent      = 0.5em,
        },
    }
\else
    \ctexset{%
        chapter = {%
            name        = {第,章},
            number      = \chinese{chapter},
            format      = \cqut@chapfont,
            beforeskip  = 12 pt,
            afterskip   = 46 pt,
            fixskip     = true,
            aftername   = \qquad,
        },
        section = {%
            format      = \cqut@secfont,
            aftername   = \quad,
            beforeskip  = 26 pt,
            afterskip   = 28 pt,
            fixskip     = true,
        },
        subsection = {%
            format      = \cqut@ssecfont,
            beforeskip  = 3 pt,
            afterskip   = 3 pt,
        },
        subsubsection = {%
            format      = \cqut@sssecfont,
            beforeskip  = 3 pt,
            afterskip   = 3 pt,
        },
    }
\fi
\setlist{%
    leftmargin = 2em,
    nosep,
}
\setlist[enumerate,1]{%
    label    = \lower 0.1em\hbox{\large{\ding{\numexpr191+\value{enumi}}}},
    labelsep = 2pt,
}
\setlist[enumerate,2]{%
    labelsep = 6pt,
}
\setlist[description]{%
    labelsep = 1em,
}
\unimathsetup{
    math-style  = ISO,
    bold-style  = ISO,
    partial     = upright,
}
\let\mathbb\relax
\DeclareMathAlphabet{\mathbb}{U}{msb}{m}{n}%
\newtheoremstyle{cqutthm}%  name
{3pt}%                      Space above
{3pt}%                      Space below
{}%                         Body font
{}%                         Indent amount
{\bfseries}%                Theorem head font
{:}%                        Punctuation after theorem head
{.5em}%                     Space after theorem head
{}%                         Theorem head spec
\theoremstyle{cqutthm}
\newtheorem{axiom}{公理}[chapter]
\newtheorem{theorem}{定理}[chapter]
\newtheorem{corollary}{推论}[chapter]
\newtheorem{remark}{注解}[chapter]
\newtheorem{assumption}{假设}[chapter]
\newtheorem{definition}{定义}[chapter]
\newtheorem{property}{性质}[chapter]
\newtheorem{proposition}{命题}[chapter]
\newtheorem{lemma}{引理}[chapter]
\renewenvironment{proof}[1][\proofname]{\par
    \pushQED{\qed}%
    \normalfont\topsep6\p@\@plus6\p@\relax
    \trivlist
    \item\relax
    {#1\@addpunct{.}}\hspace\labelsep\ignorespaces
}{\popQED\endtrivlist\@endpefalse}
\sisetup{%
    list-final-separator = { 和 },
    list-pair-separator = { 和 },
    range-phrase = {\,$\sim$\,},
}
\ifcqut@bachelor
    \newfloat{dfigure}{htbp}{lodf}[chapter]
    \floatname{dfigure}{\cqut@dfigurename}
\else
    \DeclareCaptionOption{english}[]{
        \renewcommand\figurename{\cqut@bifigure@name}
        \renewcommand\tablename{\cqut@bitable@name}
    }
    \captionsetup[bi-second]{english}
\fi
\ifcqut@bachelor\relax\else
    \AtBeginEnvironment{tabular}{\cqut@floatfont}
    \AtBeginEnvironment{tabularx}{\cqut@floatfont}
\fi
\let\oldlongtable\longtable
\let\endoldlongtable\endlongtable
\def\longtable{\zihao{5}\oldlongtable}
\AtEndEnvironment{longtable}{\normalsize}
\graphicspath{{figure/}}
\renewcommand{\captionfont}{\cqut@float@captionfont}
\renewcommand{\captionlabelfont}{\cqut@float@labelfont}
\DeclareCaptionLabelSeparator{capspace}{\quad}
\captionsetup{labelsep=capspace}
\captionsetup[table]{%
    position  = top,
    aboveskip = 0.5em,
    belowskip = 0.5em,
}
\captionsetup[figure]{%
    position  = bottom,
    aboveskip = 0.5em,
    belowskip = -0.5em,
}
\captionsetup[dfigure]{%
    position  = bottom,
    aboveskip = 1em,
    belowskip = 1em,
}
\AtBeginDocument{%
    \renewcommand{\thesubtable}{(\alph{subtable})}
    \renewcommand{\thetable}{\thechapter-\arabic{table}}
    \renewcommand{\thesubfigure}{(\alph{subfigure})}
    \renewcommand{\thefigure}{\thechapter-\arabic{figure}}
    \ifcqut@bachelor
        \renewcommand{\thedfigure}{\thechapter-\arabic{dfigure}}
        \renewcommand{\theequation}{\thechapter-\arabic{equation}}
    \else
        \renewcommand{\eqref}[1]{\equationname~\textup{(\ref{#1})}}
        \renewcommand{\theequation}{\thechapter-\arabic{equation}}
    \fi
    \renewcommand{\thelstlisting}{\thechapter-\arabic{lstlisting}}
}
\DeclareRobustCommand\inlinecite{\@inlinecite}
\def\@inlinecite#1{\begingroup\let\@cite\NAT@citenum\citep{#1}\endgroup}
\ifcqut@bachelor\relax\else
    \xpatchcmd{\dotfill}{.44em}{.3em}{}{}
    \renewenvironment{equation}{
        \@beginparpenalty\predisplaypenalty
        \@endparpenalty\postdisplaypenalty
        \refstepcounter{equation}%
        \trivlist \item[]\leavevmode
        \hb@xt@\linewidth\bgroup $\m@th% $
        \displaystyle\hfill}
        {$\;\dotfill % $
        \displaywidth\linewidth\hbox{\@eqnnum}%
        \egroup
        \endtrivlist}
\fi
\lstset{%
    aboveskip        = 0.5em,
    belowskip        = 0.5em,
    tabsize          = 4,
    basicstyle       = \cqut@codefont,
    frame            = single,
    basewidth        = {.5em,.4em},
    xleftmargin      = 1.5em,
    xrightmargin     = 1.5em,
    commentstyle     = \cqut@codecomfont,
    keywordstyle     = \cqut@codekeyfont,
    backgroundcolor  = \color{lightgray!10},
    showtabs         = false,
    showspaces       = false,
    showstringspaces = false,
    captionpos       = t,
    breaklines       = true,
    numbers          = left,
    numberstyle      = \tiny,
    numbersep        = 6 pt,
    stepnumber       = 1,
    extendedchars    = false,
    escapechar       = {@*},
}
\ifcqut@bachelor
    \newenvironment{conclusion}{%
        \chapter*{\cqut@concl@title}
        \addcontentsline{toc}{chapter}{\cqut@concl@title}
    }{}
    \newenvironment{translation}{%
        \chapter*{\cqut@trans@title}
        \addcontentsline{toc}{chapter}{\cqut@trans@title}
    }{}
\else
    \g@addto@macro\appendix{\def\cqut@econtent@chapter{Appendix~}}
    \newenvironment{achievements}{%
        \cleardoublepage
        \chapter*{\cqut@achi@title}
        \markboth{\cqut@achi@title}{\cqut@achi@title}
        \addcontentsline{toc}{chapter}{\cqut@achi@title}
        \addcontentsline{toce}{chapter}{\cqut@toce@chapfont\cqut@eachi@title}
    }{}
    \newenvironment{resume}{%
        \cleardoublepage
        \chapter*{\cqut@resu@title}
        \markboth{\cqut@resu@title}{\cqut@resu@title}
        \addcontentsline{toc}{chapter}{\cqut@resu@title}
        \addcontentsline{toce}{chapter}{\cqut@toce@chapfont\cqut@eresu@title}
    }{}
\fi
\newenvironment{acknowledgement}{%
    \chapter*{\cqut@ack@title}
    \ifcqut@bachelor\relax\else
        \markboth{\cqut@ack@title}{\cqut@ack@title}
        \addcontentsline{toce}{chapter}{\cqut@toce@chapfont\cqut@eack@title}
    \fi
    \addcontentsline{toc}{chapter}{\cqut@ack@title}
}{}
\NewDocumentEnvironment{denotation}{ O{10cm} +b }{%
    \cleardoublepage
    \chapter*{\cqut@deno@title}%
    \ifcqut@bachelor
        \addcontentsline{toc}{chapter}{\cqut@deno@title}
    \else
        \cqut@pdfmark{\cqut@deno@title}{denotation}
        \markboth{\cqut@deno@title}{\cqut@deno@title}
    \fi
    \vspace{-1em}%
    \begin{oldlongtable}[c]{l@{\extracolsep{3em}}p{#1}}
        #2
}{\end{oldlongtable}}
\ifcqut@bachelor
    \setlength{\bibsep}{3pt}
\else
    \setlength{\bibsep}{0pt}
    \renewcommand\bibfont{\linespread{1.5814}\zihao{5}}
\fi
\renewcommand{\emph}[1]{\textbf{#1}}
\renewcommand{\em}{\bfsong}
\newcommand{\cqut@ftntcircled}[1]{%
    \lower 0.15em\hbox{%
        \tikz\draw (0pt, 0pt) circle (0.4 em) node {#1};%
    }%
}
\renewcommand\thefootnote{\protect\cqut@ftntcircled{\tiny\arabic{footnote}}}
\renewcommand{\@makefntext}[1]{%
    \setlength{\leftskip}{1.5\ccwd}%
    \noindent\llap{\lower 1pt \hbox{\@thefnmark}\,}#1%
}
\urlstyle{same}
\hypersetup{%
    colorlinks         = true,
    bookmarksnumbered  = true,
    pdfhighlight       = /N,
    breaklinks         = true,
    citecolor          = cyan,
}
\ifcqut@submit
    \hypersetup{hidelinks}
\fi
\ifcqut@debugger
    \AtBeginDocument{\linenumbers}
\fi
\endinput
%%
%% End of file `cqutthesis.cls'.
