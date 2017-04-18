;;; matlab.el --- major mode for MATLAB dot-m files
;;
;; Author: Matt Wette <mwette@alumni.caltech.edu>,
;;         Eric M. Ludlam <eludlam@mathworks.com>
;; Maintainer: Eric M. Ludlam <eludlam@mathworks.com>
;; Created: 04 Jan 91
;; Version: 2.2.3
;; Keywords: Matlab
;;
;; LCD Archive Entry:
;; matlab|Eric M. Ludlam|eludlam@mathworks.com|
;; Major mode for editing and debugging MATLAB dot-m files|
;; Jan-99|2.2.3|~/modes/matlab.el|
;;
;; Copyright (C) 1997-1999 Eric M. Ludlam
;; Copyright (C) 1991-1997 Matthew R. Wette
;;
;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 2, or (at your option)
;; any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with GNU Emacs; see the file COPYING.  If not, write to
;; the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.
;;
;;; Commentary:
;;
;; This major mode for GNU Emacs provides support for editing MATLAB dot-m
;; files.  It automatically indents for block structures, line continuations
;; (e.g., ...), and comments.
;;
;; Additional features include auto-fill including auto-additions of
;; ellipsis for commands, and even strings.  Block/end construct
;; highlighting as you edit.  Primitive code-verification and
;; identification.  Templates and other code editing functions.
;; Advanced symbol completion.  Code highlighting via font-lock.
;; There are many navigation commands that let you move across blocks
;; of code at different levels.
;;
;; Lastly, there is support for running Matlab in an Emacs buffer,
;; with full shell history and debugger support (when used with the db
;; commands.)  The shell can be used as an online help while editing
;; code, providing help on functions, variables, or running arbitrary
;; blocks of code from the buffer you are editing.

;;; Finding Updates:
;;
;; The latest stable version of matlab.el can be found here:
;;
;; ftp://ftp.mathworks.com/pub/contrib/emacs_add_ons/matlab.el
;;
;; Older versions of matlab.el can be found in as matlab.el.1.10.1
;; for emacsen that do not have the latest additional utilities such
;; as tempo and derived.

;;; Installation:
;;
;;   Put the this file as "matlab.el" somewhere on your load path, then
;;   add this to your .emacs or site-init.el file:
;;
;;   (autoload 'matlab-mode "matlab" "Enter Matlab mode." t)
;;   (setq auto-mode-alist (cons '("\\.m\\'" . matlab-mode) auto-mode-alist))
;;   (autoload 'matlab-shell "matlab" "Interactive Matlab mode." t)
;;
;; User Level customizations (You need not use them all):
;;   (setq matlab-indent-function t)	; if you want function bodies indented
;;   (setq matlab-verify-on-save-flag nil) ; turn off auto-verify on save
;;   (defun my-matlab-mode-hook ()
;;     (setq fill-column 76))		; where auto-fill should wrap
;;   (add-hook 'matlab-mode-hook 'my-matlab-mode-hook)
;;   (defun my-matlab-shell-mode-hook ()
;;	'())
;;   (add-hook 'matlab-shell-mode-hook 'my-matlab-shell-mode-hook)
;;
;; Please read the mode help for matlab-mode for additional
;; configuration options.
;;
;; Syntax highlighting:
;;   To get font-lock try adding this for older emacsen:
;;     (font-lock-mode 1)
;;   Or for newer versions of Emacs:
;;     (global-font-lock-mode t)
;;   To get hilit19 support try adding:
;;     (matlab-mode-hilit)
;;
;; This package requires easymenu, tempo, and derived.
;; This package will optionally use custom, shell, and gud.
;; This package supports language specific extensions in imenu, func-menu,
;;      speedbar, font-lock, and hilit19.

;;; Mailing List:
;;
;; A mailing list has been set up where beta versions of matlab.el are
;; posted, and where comments, questions, bug reports, and answers to
;; questions can be sent.
;;
;; To subscribe send email to "lists@mathworks.com" with a body of:
;;    "subscribe matlab-emacs"
;; to unsubscribe, send email with the body of: "unsubscribe matlab-emacs"

;;; Code:

(require 'easymenu)
(require 'tempo)
(require 'derived)

(defconst matlab-mode-version "2.2.3"
  "Current version of Matlab mode.")

;; From custom web page for compatibility between versions of custom:
(eval-and-compile
  (condition-case ()
      (require 'custom)
    (error nil))
  (if (and (featurep 'custom) (fboundp 'custom-declare-variable))
      nil ;; We've got what we needed
    ;; We have the old custom-library, hack around it!
    (defmacro defgroup (&rest args)
      nil)
    (defmacro custom-add-option (&rest args)
      nil)
    (defmacro defface (&rest args) nil)
    (defmacro defcustom (var value doc &rest args)
      (` (defvar (, var) (, value) (, doc))))))

;; compatibility
(if (string-match "X[Ee]macs" emacs-version)
    (progn
      (defalias 'matlab-make-overlay 'make-extent)
      (defalias 'matlab-overlay-put 'set-extent-property)
      (defalias 'matlab-delete-overlay 'delete-extent)
      (defalias 'matlab-overlay-start 'extent-start)
      (defalias 'matlab-overlay-end 'extent-end)
      (defalias 'matlab-cancel-timer 'delete-itimer)
      (defun matlab-run-with-idle-timer (secs repeat function &rest args)
	(condition-case nil
	    (start-itimer "matlab" function secs (if repeat secs nil) t)
	  (error
	   ;; If the above doesn't work, then try this old version of
	   ;; start itimer.
	   (start-itimer "matlab" function secs (if repeat secs nil)))))
      )
  (defalias 'matlab-make-overlay 'make-overlay)
  (defalias 'matlab-overlay-put 'overlay-put)
  (defalias 'matlab-delete-overlay 'delete-overlay)
  (defalias 'matlab-overlay-start 'overlay-start)
  (defalias 'matlab-overlay-end 'overlay-end)
  (defalias 'matlab-cancel-timer 'cancel-timer)
  (defalias 'matlab-run-with-idle-timer 'run-with-idle-timer)
  )

(cond ((fboundp 'point-at-bol)
       (defalias 'matlab-point-at-bol 'point-at-bol)
       (defalias 'matlab-point-at-eol 'point-at-eol))
      ;; Emacs 20.4
      ((fboundp 'line-beginning-position)
       (defalias 'matlab-point-at-bol 'line-beginning-position)
       (defalias 'matlab-point-at-eol 'line-end-position))
      (t
       (defmacro matlab-point-at-bol ()
	 (save-excursion (beginning-of-line) (point)))
       (defmacro matlab-point-at-eol ()
	 (save-excursion (end-of-line) (point)))))


;;; User-changeable variables =================================================

;; Variables which the user can change
(defgroup matlab nil
  "Matlab mode."
  :prefix "matlab-"
  :group 'languages)

(defcustom matlab-indent-level 2
  "*The basic indentation amount in `matlab-mode'."
  :group 'matlab
  :type 'integer)

(defcustom matlab-cont-level 4
  "*Basic indentation after continuation if no other methods are found."
  :group 'matlab
  :type 'integer)

(defcustom matlab-case-level '(1 . 1)
  "*How far to indent case/otherwise statements in a switch.
This can be an integer, which is the distance to indent the CASE and
OTHERWISE commands, and how far to indent commands appearing in CASE
and OTHERWISE blocks.  It can also be a cons cell which is of form
  (CASEINDENT . COMMANDINDENT)
where CASEINDENT is the indentation of the CASE and OTHERWISE
statements, and COMMANDINDENT is the indentation of commands appearing
after the CASE or OTHERWISE command."
  :group 'matlab
  :type 'sexp)

(defcustom matlab-indent-function nil
  "*If non-nil, indent body of function."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-indent-past-arg1-functions
  "[sg]et\\(_param\\)?\\|waitfor"
  "*Regex describing functions whose first arg is special.
This specialness means that all following parameters which appear on
continued lines should appear indented to line up with the second
argument, not the first argument."
  :group 'matlab
  :type 'string)

(defcustom matlab-arg1-max-indent-length 15
  "*The maximum length to indent when indenting past arg1.
If arg1 is exceptionally long, then only this number of characters
will be indented beyond the open paren starting the parameter list.")

(defcustom matlab-maximum-indents '(;; = is a convenience. Don't go too far
				    (?= . (10 . 4))
				    ;; Fns should provide hard limits
				    (?\( . 50)
				    ;; Matrix/Cell arrays
				    (?\[ . 20)
				    (?\{ . 20))
  "Alist of maximum indentations when lining up code.
Each element is of the form (CHAR . INDENT) where char is a character
the indent engine is using, and INDENT is the maximum indentation
allowed.  Indent could be of the form (MAXIMUM . INDENT), where
MAXIMUM is the maximum allowed calculated indent, and INDENT is the
amount to use if MAXIMUM is reached.")

(defcustom matlab-handle-simulink t
  "*If true, add in a few simulink customizations.
This variable's state is mostly useful when set at load time when
simulink font lock keywords can be removed.  This will handle
additional cases as the need arrises."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-auto-fill t
  "*If true, set variable `auto-fill-function' to our function at startup."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-fill-fudge 10
  "Number of characters around `fill-column' we can fudge filling.
Basically, there are places that are very convenient to fill at, but
might not be the closest fill spot, or occur after `fill-column'.
If they occur within this fudge factor, we will use them.
Also, if none of the above occur, and we find a symbol to break at,
but an open paren (group) starts or ends within this fudge factor,
move there to boost the amount of fill leverage we can get."
  :group 'matlab
  :type 'integer)

(defcustom matlab-fill-fudge-hard-maximum 79
  "The longest line allowed when auto-filling code.
This overcomes situations where the `fill-column' plus the
`matlab-fill-fudge' is greater than some hard desired limit."
  :group 'matlab
  :type 'integer)

(defcustom matlab-elipsis-string "..."
  "Text used to perform continuation on code lines.
This is used to generate and identify continuation lines.")

(defcustom matlab-fill-code t
  "*If true, `auto-fill-mode' causes code lines to be automatically continued."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-fill-count-ellipsis-flag t
  "*Non-nil means to count the ellipsis when auto filling.
This effectively shortens the `fill-column' by the length of
`matlab-elipsis-string'.")

(defcustom matlab-fill-strings-flag t
  "*Non-nil means that when auto-fill is on, strings are broken across lines.
If `matlab-fill-count-ellipsis-flag' is non nil, this shortens the
`fill-column' by the length of `matlab-elipsis-string'.")

(defcustom matlab-comment-column 40
  "*The goal comment column in `matlab-mode' buffers."
  :group 'matlab
  :type 'integer)

(defcustom matlab-comment-line-s "% "
  "*String to start comment on line by itself."
  :group 'matlab
  :type 'string)

(defcustom matlab-comment-on-line-s "% "
  "*String to start comment on line with code."
  :group 'matlab
  :type 'string)

(defcustom matlab-comment-region-s "% $$$ "
  "*String inserted by \\[matlab-comment-region] at start of each line in \
region."
  :group 'matlab
  :type 'string)

(defcustom matlab-verify-on-save-flag t
  "*Non-nil means to verify M whenever we save a file."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-mode-verify-fix-functions
  '(matlab-mode-vf-functionname
    matlab-mode-vf-block-matches-forward
    matlab-mode-vf-block-matches-backward)
  "List of function symbols which perform a verification and fix to M code.
Each function gets no arguments, and returns nothing.  They can move
point, but it will be restored for them."
  :group 'matlab
  :type '(repeat (choice :tag "Function: "
			 '(matlab-mode-vf-functionname
			   matlab-mode-vf-block-matches-forward
			   matlab-mode-vf-block-matches-backward
			   matlab-mode-vf-quietafy-buffer
			   ))))

(defcustom matlab-block-verify-max-buffer-size 50000
  "*Largest buffer size allowed for block verification during save."
  :group 'matlab
  :type 'integer)

(defcustom matlab-vers-on-startup t
  "*If non-nil, show the version number on startup."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-highlight-block-match-flag t
  "*Non-nil means to highlight the matching if/end/whatever.
The highlighting only occurs when the cursor is on a block start or end
keyword."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-show-periodic-code-details-flag nil
  "*Non-nil means to show code details in the minibuffer.
This will only work if `matlab-highlight-block-match-flag' is non-nil."
  :group 'matlab
  :type 'boolean)

(defcustom matlab-mode-hook nil
  "*List of functions to call on entry to Matlab mode."
  :group 'matlab
  :type 'hook)

(defcustom matlab-completion-technique 'complete
  "*How the `matlab-complete-symbol' interfaces with the user.
Valid values are:

'increment - which means that new strings are tried with each
             successive call until all methods are exhausted.
             (Similar to `hippie-expand'.)
'complete  - Which means that if there is no single completion, then
             all possibilities are displayed in a completion buffer."
  :group 'matlab
  :type '(radio (const :tag "Incremental completion (hippie-expand)."
		       increment)
		(const :tag "Show completion buffer."
		       complete)))

;; Load in the region we use for highlighting stuff.
(if (and (featurep 'custom) (fboundp 'custom-declare-variable))

    (let ((l-region-face (if (facep 'region) 'region 'zmacs-region)))
      ;; If we have custom, we can make our own special face like this
      (defface matlab-region-face
	(list
	 (list t
	       (list :background (face-background l-region-face)
		     :foreground (face-foreground l-region-face))))
	"*Face used to highlight a matlab region."
	:group 'matlab))

  ;; If we do not, then we can fake it by copying 'region.
  (cond ((facep 'region)
	 (copy-face 'region 'matlab-region-face))
	(t
	 (copy-face 'zmacs-region 'matlab-region-face))))

(defvar matlab-unterminated-string-face 'matlab-unterminated-string-face
  "Self reference for unterminated string face.")

(defvar matlab-simulink-keyword-face 'matlab-simulink-keyword-face
  "Self reference for simulink keywords.")

(defun matlab-font-lock-adjustments ()
  "Make adjustments for font lock.
If font lock is not loaded, lay in wait."
  (if (and (featurep 'custom) (fboundp 'custom-declare-variable))
	
      (progn
	(defface matlab-unterminated-string-face
	  (list
	   (list t
		 (list :background (face-background font-lock-string-face)
		       :foreground (face-foreground font-lock-string-face)
		       :underline t)))
	  "*Face used to highlight unterminated strings."
	  :group 'matlab)
	(defface matlab-simulink-keyword-face
	  (list
	   (list t
		 (list :background (face-background font-lock-type-face)
		       :foreground (face-foreground font-lock-type-face)
		       :underline t)))
	  "*Face used to highlight simulink specific functions.")
	)
      
    ;; Now, lets make the unterminated string face
    (cond ((facep 'font-lock-string-face)
	   (copy-face 'font-lock-string-face
		      'matlab-unterminated-string-face))
	  (t
	   (make-face 'matlab-unterminated-string-face)))
    (set-face-underline-p 'matlab-unterminated-string-face t)
    
    ;; Now make some simulink faces
    (cond ((facep 'font-lock-type-face)
	   (copy-face 'font-lock-type-face 'matlab-simulink-keyword-face))
	  (t
	   (make-face 'matlab-simulink-keyword-face)))
    (set-face-underline-p 'matlab-simulink-keyword-face t)
    )
  (remove-hook 'font-lock-mode-hook 'matlab-font-lock-adjustments))

;; Make the adjustments for font lock after it's loaded.
;; I found that eval-after-load was unreliable.
(if (featurep 'font-lock)
    (matlab-font-lock-adjustments)
  (add-hook 'font-lock-mode-hook 'matlab-font-lock-adjustments))


;;; Matlab mode variables =====================================================

(defvar matlab-tempo-tags nil
  "List of templates used in Matlab mode.")

;; syntax table
(defvar matlab-mode-syntax-table
  (let ((st (make-syntax-table (standard-syntax-table))))
    (modify-syntax-entry ?_  "_" st)
    (modify-syntax-entry ?%  "<" st)
    (modify-syntax-entry ?\n ">" st)
    (modify-syntax-entry ?\\ "." st)
    (modify-syntax-entry ?\t " " st)
    (modify-syntax-entry ?+  "." st)
    (modify-syntax-entry ?-  "." st)
    (modify-syntax-entry ?*  "." st)
    (modify-syntax-entry ?'  "." st)
    (modify-syntax-entry ?/  "." st)
    (modify-syntax-entry ?=  "." st)
    (modify-syntax-entry ?<  "." st)
    (modify-syntax-entry ?>  "." st)
    (modify-syntax-entry ?&  "." st)
    (modify-syntax-entry ?|  "." st)
    st)
  "The syntax table used in `matlab-mode' buffers.")

(defvar matlab-mode-special-syntax-table
  (let ((st (copy-syntax-table matlab-mode-syntax-table)))
    ;; Make _ a part of words so we can skip them better
    (modify-syntax-entry ?_  "w" st)
    st)
  "The syntax table used when navigating blocks.")

;; abbrev table
(defvar matlab-mode-abbrev-table nil
  "The abbrev table used in `matlab-mode' buffers.")

(define-abbrev-table 'matlab-mode-abbrev-table ())

;;; Keybindings ===============================================================

(defvar matlab-help-map
  (let ((km (make-sparse-keymap)))
    (define-key km "r" 'matlab-shell-run-command)
    (define-key km "f" 'matlab-shell-describe-command)
    (define-key km "a" 'matlab-shell-apropos)
    (define-key km "v" 'matlab-shell-describe-variable)
    (define-key km "t" 'matlab-shell-topic-browser)
    km)
  "The help key map for `matlab-mode' and `matlab-shell-mode'.")

(defvar matlab-insert-map
  (let ((km (make-sparse-keymap)))
    (define-key km "c" 'matlab-insert-next-case)
    (define-key km "e" 'matlab-insert-end-block)
    (define-key km "i" 'tempo-template-matlab-if)
    (define-key km "I" 'tempo-template-matlab-if-else)
    (define-key km "f" 'tempo-template-matlab-for)
    (define-key km "s" 'tempo-template-matlab-switch)
    (define-key km "t" 'tempo-template-matlab-try)
    (define-key km "w" 'tempo-template-matlab-while)
    (define-key km "F" 'tempo-template-matlab-function)
    (define-key km "'" 'matlab-stringify-region)
    ;; Not really inserts, but auto coding stuff
    (define-key km "\C-s" 'matlab-ispell-strings)
    (define-key km "\C-c" 'matlab-ispell-comments)
    km)
  "Keymap used for inserting simple texts based on context.")

;; mode map
(defvar matlab-mode-map
  (let ((km (make-sparse-keymap)))
    (define-key km [return] 'matlab-return)
    (define-key km "\C-c;" 'matlab-comment-region)
    (define-key km "\C-c:" 'matlab-uncomment-region)
    (define-key km [(control c) return] 'matlab-comment-return)
    (define-key km [(control c) (control c)] matlab-insert-map)
    (define-key km [(control c) (control f)] 'matlab-fill-comment-line)
    (define-key km [(control c) (control j)] 'matlab-justify-line)
    (define-key km [(control c) (control q)] 'matlab-fill-region)
    (define-key km [(control c) (control s)] 'matlab-shell-save-and-go)
    (define-key km [(control c) (control r)] 'matlab-shell-run-region)
    (define-key km [(control c) (control t)] 'matlab-show-line-info)
    (define-key km [(control c) ?. ] 'matlab-find-file-on-path)
    (define-key km [(control h) (control m)] matlab-help-map)
    (define-key km [(control j)] 'matlab-linefeed)
    (define-key km "\M-\r" 'newline)
    (define-key km [(meta \;)] 'matlab-comment)
    (define-key km [(meta q)] 'matlab-fill-paragraph)
    (define-key km [(meta a)] 'matlab-beginning-of-command)
    (define-key km [(meta e)] 'matlab-end-of-command)
    (define-key km [(meta j)] 'matlab-comment-line-break-function)
    (define-key km [(meta s)] 'matlab-show-matlab-shell-buffer)
    (define-key km "\M-\t" 'matlab-complete-symbol)
    (define-key km [(meta control f)] 'matlab-forward-sexp)
    (define-key km [(meta control b)] 'matlab-backward-sexp)
    (define-key km [(meta control a)] 'matlab-beginning-of-defun)
    (define-key km [(meta control e)] 'matlab-end-of-defun)
    (if (string-match "XEmacs" emacs-version)
	(define-key km [(control meta button1)] 'matlab-find-file-click)
      (define-key km [(control meta mouse-2)] 'matlab-find-file-click))
    (substitute-key-definition 'comment-region 'matlab-comment-region
			       km) ; global-map ;torkel
    km)
  "The keymap used in `matlab-mode'.")

;;; Font locking keywords =====================================================

(defvar matlab-string-start-regexp "\\(^\\|[^]})a-zA-Z0-9_.']\\)"
  "Regexp used to represent the character before the string char '.
The ' character has restrictions on what starts a string which is needed
when attempting to understand the current context.")

;; To quote a quote, put two in a row, thus we need an anchored
;; first quote.  In addition, we don't want to color strings in comments.
(defvar matlab-string-end-regexp "[^'\n]*\\(''[^'\n]*\\)*'"
  "Regexp used to represent the character pattern for ending a string.
The ' character can be used as a transpose, and can transpose transposes.
Therefore, to end, we must check all that goop.")

(defun matlab-font-lock-string-match-normal (limit)
  "When font locking strings, call this function for normal strings.
Argument LIMIT is the maximum distance to scan."
  (matlab-font-lock-string-match-here
   (concat matlab-string-start-regexp
	   "\\('" matlab-string-end-regexp "\\)"
	   "\\([^']\\|$\\)")
   limit))

(defun matlab-font-lock-string-match-unterminated (limit)
  "When font locking strings, call this function for normal strings.
Argument LIMIT is the maximum distance to scan."
  (matlab-font-lock-string-match-here
   (concat matlab-string-start-regexp "\\('[^'\n]*\\(''[^'\n]*\\)*\\)$")
   limit))

(defun matlab-font-lock-string-match-here (regex limit)
  "When font-locking strings, call this function to determine a match.
Argument REGEX is the expression to scan for.  Match 2 must be the string.
Argument LIMIT is the maximum distance to scan."
  (let (e)
    (while (and (re-search-forward regex limit t)
		(progn
		  ;; This gets us out of a comment after the string.
		  (setq e (match-end 2))
		  (goto-char (match-beginning 2))
		  (prog1
		      (or (matlab-cursor-in-comment)
			  (if (bolp) nil
			    (save-excursion
			      (forward-char -1)
			      (matlab-cursor-in-string))))
		    (goto-char e))))
      (setq e nil))
    (if (not e)
	nil
      (goto-char e)
      t)))

(defun matlab-font-lock-comment-match (limit)
  "When font-locking comments, call this function to determine a match.
Argument LIMIT is the maximum distance to scan."
  (let (e)
    (while (and (re-search-forward "\\(%[^%\n]*\\)" limit t)
		(progn
		  (setq e (match-end 1))
		  (member (get-text-property (match-beginning 0) 'face)
			  '(font-lock-string-face
			    matlab-unterminated-string-face))))
      (setq e nil))
    (if (not e)
	nil
      (goto-char e)
      t)))

(defun matlab-find-unreachable-code (limit)
  "Find code that is if'd out with if(0), and mark it as a comment.
The if(0) and else/end construct sould be highlighted differently.
Argument LIMIT is the maximum distance to search."
  (if (and (< (point) limit)
	   (re-search-forward
	    "\\<\\(if\\>\\s-*(?\\s-*0\\s-*)?\\)"
	    limit t))
      (let ((b1 (match-beginning 1))
	    (e1 (match-end 1))
	    (b2 nil) (e2 nil)
	    (b3 nil) (e3 nil))
	(goto-char b1)
	(condition-case nil
	    (progn
	      ;; Go forward over the matlab sexp.  Include scanning
	      ;; for ELSE since parts of the ELSE block are not
	      ;; `commented out'.
	      (matlab-forward-sexp t)
	      (forward-word -1)
	      ;; Is there an ELSE in this block?
	      (if (looking-at (matlab-block-mid-re))
		  (progn
		    (setq b3 (match-beginning 0)
			  e3 (match-end 0))
		    ;; Now find the REAL end.
		    (matlab-forward-sexp)
		    (forward-word -1)))
	      ;; End of block stuff
	      (if (looking-at (matlab-block-end-re))
		  (progn
		    (setq b2 (match-beginning 0)
			  e2 (match-end 0))
		    ;; make sure something exists...
		    (if (not b3) (setq b3 b2 e3 e2)))
		(error "Eh?"))
	      ;; Ok, build up some match data.
	      (set-match-data
	       (list b1 e2		;the real deal.
		     b1 e1		;if (0)
		     b2 e2		;end
		     b3 e3		;else (if applicable.)
		     b1 e3))		;body commented out.
	      t)
	  (error nil)))))

(defcustom matlab-keyword-list '("global" "for" "while" "if" "elseif" "else"
				 "endfunction" "return" "break"
				 "switch" "case" "otherwise" "try"
				 "catch")
  "List of keywords for Matlab used in highlighting.
Customizing this variable is only useful if `regexp-opt' is available."
  :group 'matlab
  :type '(repeat (string :tag "Keyword: ")))

(defcustom matlab-handle-graphics-list '("figure" "axes" "axis" "line"
					"surface" "patch" "text" "light"
					"image" "set" "get" "uicontrol"
					"uimenu" "uitoolbar"
					"uitoggletool" "uipushtool"
					"uicontext" "uicontextmenu"
					"setfont" "setcolor")
  "List of handle graphics functions used in highlighting.
Customizing this variable is only useful if `regexp-opt' is available."
  :group 'matlab
  :type '(repeat (string :tag "HG Keyword: ")))

(defcustom matlab-debug-list '("dbstop" "dbclear" "dbcont" "dbdown" "dbmex"
			      "dbstack" "dbstatus" "dbstep" "dbtype" "dbup"
			      "dbquit")
  "List of debug commands used in highlighting.
Customizing this variable is only useful if `regexp-opt' is available."
  :group 'matlab
  :type '(repeat (string :tag "Debug Keyword: ")))

;; font-lock keywords
(defvar matlab-font-lock-keywords
  (list
   ;; String quote chars are also used as transpose, but only if directly
   ;; after characters, numbers, underscores, or closing delimiters.
   '(matlab-font-lock-string-match-normal 2 font-lock-string-face)
   ;; A string with no termination is not currently highlighted.
   ;; This will show that the string needs some attention.
   '(matlab-font-lock-string-match-unterminated
     2 matlab-unterminated-string-face)
   ;; Comments must occur after the string, that way we can check to see
   ;; if the comment start char has occurred inside our string. (EL)
   '(matlab-font-lock-comment-match 1 font-lock-comment-face)
   ;; General keywords
   (list
    (if (fboundp 'regexp-opt)
	(concat "\\<\\(" (regexp-opt matlab-keyword-list) "\\)\\>")
      ;; Original hard-coded value for pre Emacs 20.1
      "\\<\\(break\\|ca\\(se\\|tch\\)\\|e\\(lse\\(\\|if\\)\\|ndfunction\\)\
\\|for\\|global\\|if\\|otherwise\\|return\\|switch\\|try\\|while\\)\\>")
    '(0 font-lock-keyword-face))
   ;; The end keyword is only a keyword when not used as an array
   ;; dereferencing part.
   '("\\(^\\|[;,]\\)[ \t]*\\(end\\)\\b"
     2 (if (matlab-valid-end-construct-p) font-lock-keyword-face nil))
   ;; How about unreachable code?  MUsT BE AFTER KEYWORDS in order to
   ;; get double-highlighting.
   '(matlab-find-unreachable-code
     (1 'underline prepend)		;if part
     (2 'underline prepend)		;end part
     (3 'underline prepend)		;else part (if applicable)
     (4 font-lock-comment-face prepend)	;commented out part.
     )
   ;; The global keyword defines some variables.  Mark them.
   '("^\\s-*global\\s-+"
     ("\\(\\w+\\)\\(\\s-*=[^,; \t\n]+\\|[, \t;]+\\|$\\)"
      nil nil (1 font-lock-variable-name-face)))
   ;; Handle graphics stuff
   (list
    (if (fboundp 'regexp-opt)
	(concat "\\<\\(" (regexp-opt matlab-handle-graphics-list) "\\)\\>")
      ;; The original regular expression for pre Emacs 20.1
      "\\<\\(ax\\(es\\|is\\)\\|figure\\|get\\|image\\|li\\(ght\\|ne\\)\\|\
patch\\|s\\(et\\(\\|color\\|font\\)\\|urface\\)\\|text\\|\
ui\\(cont\\(ext\\(\\|menu\\)\\|rol\\)\\|menu\\|\
\\(toggle\\|push\\)tool\\|toolbar\\)\\)\\>")
    '(0 font-lock-type-face))
   )
  "Expressions to highlight in Matlab mode.")

(defvar matlab-gaudy-font-lock-keywords
  (append
   matlab-font-lock-keywords
   (list
    ;; defining a function, a (possibly empty) list of assigned variables,
    ;; function name, and an optional (possibly empty) list of input variables
    (list (concat "^\\s-*\\(function\\)\\>[ \t\n.]*"
		  "\\(\\[[^]]*\\]\\|\\sw+\\)[ \t\n.]*"
		  "=[ \t\n.]*\\(\\sw+\\)[ \t\n.]*"
		  "\\(([^)]*)\\)?\\s-*\\([,;\n%]\\|$\\)")
	  '(1 font-lock-keyword-face append)
	  '(2 font-lock-variable-name-face append)
	  '(3 font-lock-function-name-face append))
    ;; defining a function, a function name, and an optional (possibly
    ;; empty) list of input variables
    (list (concat "^\\s-*\\(function\\)[ \t\n.]+"
		  "\\(\\sw+\\)[ \t\n.]*"
		  "\\(([^)]*)\\)?\\s-*\\([,;\n%]\\|$\\)")
	  '(1 font-lock-keyword-face append)
	  '(2 font-lock-function-name-face append))
    ;; Anchor on the function keyword, highlight params
    (list (concat "^\\s-*function\\>[ \t\n.]*"
		  "\\(\\(\\[[^]]*\\]\\|\\sw+\\)[ \t\n.]*=[ \t\n.]*\\)?"
		  "\\sw+\\s-*(")
	  '("\\s-*\\(\\sw+\\)\\s-*[,)]" nil nil
	    (1 font-lock-variable-name-face)))
    ;; I like variables for FOR loops
    '("\\<\\(for\\)\\s-+\\(\\sw+\\)\\s-*=\\s-*\
\\(\\([^\n,;%(]+\\|([^\n%)]+)\\)+\\)"
      (1 font-lock-keyword-face)
      (2 font-lock-variable-name-face append)
      (3 font-lock-reference-face append))
    ;; Items after a switch statements are cool
    '("\\<\\(case\\|switch\\)\\s-+\\({[^}\n]+}\\|[^,%\n]+\\)"
      (1 font-lock-keyword-face) (2 font-lock-reference-face))
    ;; How about a few matlab constants such as pi, infinity, and sqrt(-1)?
    ;; The ^>> is in case I use this in an interactive mode someday
    '("\\<\\(eps\\|pi\\|inf\\|Inf\\|NaN\\|nan\\|ans\\|i\\|j\\|^>>\\)\\>"
      1 font-lock-reference-face)
    '("[0-9]\\.?\\(i\\|j\\)" 1 font-lock-reference-face)
    ;; Define these as variables since this is about as close
    ;; as matlab gets to variables
    (list (concat "\\<" matlab-indent-past-arg1-functions "\\s-*")
	  '("(\\s-*\\(\\w+\\)\\s-*\\(,\\|)\\)" nil nil
	    (1 font-lock-variable-name-face)))
    ))
  "Expressions to highlight in Matlab mode.")

(defvar matlab-really-gaudy-font-lock-keywords
  (append
   matlab-gaudy-font-lock-keywords
   (list
    ;; Since it's a math language, how bout dem symbols?
    '("\\([<>~]=?\\|\\.[*^']\\|==\\|\\<xor\\>\\|[-!^&|*+\\/~:]\\)"
      1 font-lock-type-face)
    ;; How about references in the HELP text.
    (list (concat "^" matlab-comment-line-s "\\s-*"
		  "\\(\\([A-Z]+\\s-*=\\s-+\\|\\[[^]]+]\\s-*=\\s-+\\|\\)"
		  "\\([A-Z][0-9A-Z]+\\)\\(([^)\n]+)\\| \\)\\)")
	  '(1 font-lock-reference-face prepend))
    (list (concat "^" matlab-comment-line-s "\\s-*"
		  "See also\\s-+")
	  '("\\([A-Z][A-Z0-9]+\\)\\([,.]\\| and\\|$\\) *" nil nil
	    (1 font-lock-reference-face prepend)))
    (list (concat "^" matlab-comment-line-s "\\s-*"
		  "\\(\\$" "Revision" "[^\n$]+\\$\\)")
	  '(1 font-lock-reference-face prepend))
    ;; continuation ellipsis.
    '("[^.]\\(\\.\\.\\.+\\)\\([^\n]*\\)" (1 'underline)
      (2 font-lock-comment-face))
    ;; How about debugging statements?
    ;;'("\\<\\(db\\sw+\\)\\>" 1 'bold)
    (list
     (if (fboundp 'regexp-opt)
	 (concat "\\<\\(" (regexp-opt matlab-debug-list) "\\)\\>")
       ;; pre-regexp-opt days.
       "\\<\\(db\\(c\\(lear\\|ont\\)\\|down\\|mex\\|quit\\|\
st\\(a\\(ck\\|tus\\)\\|ep\\|op\\)\\|type\\|up\\)\\)\\>")
     '(0 'bold)))
   (if matlab-handle-simulink
       ;; Simulink functions, but only if the user wants it.
       (list (list (concat "\\<\\(\\([sg]et_param\\|sim\\([gs]et\\)?\\|"
			   "\\(mld\\|ss\\)[A-Z]\\w+\\)\\|"
			   "\\(new\\|open\\|close\\|save\\|find\\)_system\\|"
			   "\\(add\\|delete\\|replace\\)_\\(block\\|line\\)\\|"
			   "simulink\\|bd\\(root\\|close\\)"
			   "\\)\\>")
		   1 matlab-simulink-keyword-face))
     nil))
  "Expressions to highlight in Matlab mode.")

(defvar matlab-shell-font-lock-keywords
  (list
   ;; How about Errors?
   '("^\\(Error in\\|Syntax error in\\)\\s-+==>\\s-+\\(.+\\)$"
     (1 font-lock-comment-face) (2 font-lock-string-face))
   ;; and line numbers
   '("^\\(On line [0-9]+\\)" 1 font-lock-comment-face)
   ;; User beep things
   '("\\(\\?\\?\\?[^\n]+\\)" 1 font-lock-comment-face)
   ;; Useful user commands, but not useful programming constructs
   '("\\<\\(demo\\|whatsnew\\|info\\|subscribe\\|help\\|doc\\|lookfor\\|what\
\\|whos?\\|cd\\|clear\\|load\\|save\\|helpdesk\\|helpwin\\)\\>"
     1 font-lock-keyword-face)
   ;; Various notices
   '(" M A T L A B (R) " 0 'underline)
   '("All Rights Reserved" 0 'italic)
   '("\\((c)\\s-+Copyright[^\n]+\\)" 1 font-lock-comment-face)
   '("\\(Version\\)\\s-+\\([^\n]+\\)"
     (1 font-lock-function-name-face) (2 font-lock-variable-name-face))
   )
  "Additional keywords used by Matlab when reporting errors in interactive\
mode.")


;; hilit19 patterns
(defvar matlab-hilit19-patterns
  '(
    ("\\(^\\|[^%]\\)\\(%[ \t].*\\|%\\)$" 2 comment)
    ("\\(^\\|[;,]\\)[ \t]*\\(\
function\\|global\\|for\\|while\\|if\\|elseif\\|else\\|end\\(function\\)?\
\\|return\\|switch\\|case\\|otherwise\\|try\\|catch\\)\\b" 2 keyword)))

;; func-menu support for matlab
(defvar fume-function-name-regexp-matlab
  "^\\s-*function\\>[ \t\n.]*\\(\\(\\[[^]]*\\]\\|\\sw+\\)[ \t\n.]*\
=\[ \t\n.]*\\)?\\([a-zA-Z0-9_]+\\)"
  "Expression to get matlab function/procedure names.")

;; "The connection between a mode and the regexp that matches function
(if (boundp  'fume-function-name-regexp-alist)
    (add-to-list 'fume-function-name-regexp-alist
		 '(matlab-mode . fume-function-name-regexp-matlab)))

;;; Specialised routine to find the next Matlab function or subroutine.
(defun fume-find-next-matlab-function-name (buffer)
  "Searches for the next matlab function in BUFFER."
  (fume-find-next-sexp buffer))

(if (boundp  'fume-function-name-regexp-alist)
    (add-to-list 'fume-find-function-name-method-alist
		 '(matlab-mode . fume-find-next-matlab-function-name)))

;; Imenu support.  Recycle the expression for fume since they are the
;; same, and this reduced code duplication.
(defvar matlab-imenu-generic-expression
  `((nil ,fume-function-name-regexp-matlab 3))
  "Expressions which find function headings in Matlab M files.")


;;; Matlab mode entry point ==================================================

;;;###autoload
(defun matlab-mode ()
  "Matlab-mode is a major mode for editing MATLAB dot-m files.
\\<matlab-mode-map>
Convenient editing commands are:
 \\[matlab-comment-region]   - Comment/Uncomment out a region of code.
 \\[matlab-fill-comment-line] - Fill the current comment line.
 \\[matlab-fill-region] - Fill code and comments in region.
 \\[matlab-fill-paragraph]     - Refill the current command or comment.
 \\[matlab-complete-symbol]   - Symbol completion of matlab symbols\
based on the local syntax.

Convenient navigation commands are:
 \\[matlab-beginning-of-command]   - Move to the beginning of a command.
 \\[matlab-end-of-command]   - Move to the end of a command.
 \\[matlab-beginning-of-defun] - Move to the beginning of a function.
 \\[matlab-end-of-defun] - Move do the end of a function.
 \\[matlab-forward-sexp] - Move forward over a syntactic block of code.
 \\[matlab-backward-sexp] - Move backwards over a syntactic block of code.

Convenient template insertion commands:
 \\[tempo-template-matlab-function] - Insert a function definition.
 \\[tempo-template-matlab-if] - Insert an IF END block.
 \\[tempo-template-matlab-for] - Insert a FOR END block.
 \\[tempo-template-matlab-switch] - Insert a SWITCH END statement.
 \\[matlab-insert-next-case] - Insert the next CASE condition in a SWITCH.
 \\[matlab-insert-end-block] - Insert a matched END statement.  With \
optional ARG, reindent.
 \\[matlab-stringify-region] - Convert some plaintext into a string \
with correctly quoted chars.

Variables:
  `matlab-indent-level'		Level to indent blocks.
  `matlab-cont-level'		Level to indent continuation lines.
  `matlab-case-level'		Level to unindent case statements.
  `matlab-indent-past-arg1-functions'
                                Regexp of functions to indent past the first
                                  argument on continuation lines.
  `matlab-maximum-indents'      List of maximum indents during lineups.
  `matlab-comment-column'       Goal column for on-line comments.
  `fill-column'			Column used in auto-fill.
  `matlab-indent-function'	If non-nil, indents body of MATLAB functions.
  `matlab-return-function'	Customize RET handling with this function
  `matlab-auto-fill'            Non-nil, do auto-fill at startup
  `matlab-fill-code'            Non-nil, auto-fill code.
  `matlab-fill-strings'         Non-nil, auto-fill strings.
  `matlab-verify-on-save-flag'  Non-nil, enable code checks on save
  `matlab-highlight-block-match-flag'
                                Enable matching block begin/end keywords
  `matlab-vers-on-startup'	If t, show version on start-up.
  `matlab-handle-simulink'      If t, enable simulink keyword highlighting.

All Key Bindings:
\\{matlab-mode-map}"
  (interactive)
  (kill-all-local-variables)
  (use-local-map matlab-mode-map)
  (setq major-mode 'matlab-mode)
  (setq mode-name "Matlab")
  (setq local-abbrev-table matlab-mode-abbrev-table)
  (set-syntax-table matlab-mode-syntax-table)
  (make-local-variable 'indent-line-function)
  (setq indent-line-function 'matlab-indent-line)
  (make-local-variable 'paragraph-start)
  (setq paragraph-start (concat "^$\\|" page-delimiter))
  (make-local-variable 'paragraph-separate)
  (setq paragraph-separate paragraph-start)
  (make-local-variable 'paragraph-ignore-fill-prefix)
  (setq paragraph-ignore-fill-prefix t)
  (make-local-variable 'comment-start-skip)
  (setq comment-start-skip "%\\s-+")
  (make-local-variable 'comment-start)
  (setq comment-start "%")
  (make-local-variable 'comment-column)
  (setq comment-column matlab-comment-column)
  (make-local-variable 'comment-indent-function)
  (setq comment-indent-function 'matlab-comment-indent)
  (make-local-variable 'fill-column)
  (setq fill-column default-fill-column)
  (make-local-variable 'auto-fill-function)
  (if matlab-auto-fill (setq auto-fill-function 'matlab-auto-fill))
  ;; Emacs 20 supports this variable.  This lets users turn auto-fill
  ;; on and off and still get the right fill function.
  (make-local-variable 'normal-auto-fill-function)
  (setq normal-auto-fill-function 'matlab-auto-fill)
  (make-local-variable 'fill-prefix)
  (make-local-variable 'imenu-generic-expression)
  (setq imenu-generic-expression matlab-imenu-generic-expression)
  ;; Save hook for verifying src.  This lets us change the name of
  ;; the function in `write-file' and have the change be saved.
  ;; It also lets us fix mistakes before a `save-and-go'.
  (make-local-variable 'write-contents-hooks)
  (add-hook 'write-contents-hooks 'matlab-mode-verify-fix-file-fn)
  ;; Tempo tags
  (make-local-variable 'tempo-local-tags)
  (setq tempo-local-tags (append matlab-tempo-tags tempo-local-tags))
  ;; give each file it's own parameter history
  (make-local-variable 'matlab-shell-save-and-go-history)
  (make-local-variable 'font-lock-defaults)
  (setq font-lock-defaults '((matlab-font-lock-keywords
			      matlab-gaudy-font-lock-keywords
			      matlab-really-gaudy-font-lock-keywords
			      )
			     t ; do not do string/comment highlighting
			     nil ; keywords are case sensitive.
			     ;; This puts _ as a word constituent,
			     ;; simplifying our keywords significantly
			     ((?_ . "w"))))
  (matlab-enable-block-highlighting 1)
  (if window-system (matlab-frame-init))
  (run-hooks 'matlab-mode-hook)
  (if matlab-vers-on-startup (matlab-show-version)))

;;; Utilities =================================================================

(defun matlab-show-version ()
  "Show the version number in the minibuffer."
  (interactive)
  (message "matlab-mode, version %s" matlab-mode-version))

(defun matlab-find-prev-line ()
  "Recurse backwards until a code line is found."
  (if (= -1 (forward-line -1)) nil
    (if (or (matlab-ltype-empty)
	    (matlab-ltype-comm-ignore))
	(matlab-find-prev-line) t)))

(defun matlab-prev-line ()
  "Go to the previous line of code.  Return nil if not found."
  (interactive)
  (let ((old-point (point)))
    (if (matlab-find-prev-line) t (goto-char old-point) nil)))

(defun matlab-uniquafy-list (lst)
  "Return a list that is a subset of LST where all elements are unique."
  (let ((nlst nil))
    (while lst
      (if (and (car lst) (not (member (car lst) nlst)))
	  (setq nlst (cons (car lst) nlst)))
      (setq lst (cdr lst)))
    (nreverse nlst)))

; Aki Vehtari <Aki.Vehtari@hut.fi> recommends this: (19.29 required)
;(require 'backquote)
;(defmacro matlab-navigation-syntax (&rest body)
;  "Evaluate BODY with the matlab-mode-special-syntax-table"
;  '(let	((oldsyntax (syntax-table)))
;    (unwind-protect
;	(progn
;	  (set-syntax-table matlab-mode-special-syntax-table)
;	   ,@body)
;      (set-syntax-table oldsyntax))))

(defmacro matlab-navigation-syntax (&rest forms)
  "Set the current environment for syntax-navigation and execute FORMS."
  (list 'let '((oldsyntax (syntax-table))
	       (case-fold-search nil))
	 (list 'unwind-protect
		(list 'progn
		       '(set-syntax-table matlab-mode-special-syntax-table)
			(cons 'progn forms))
		'(set-syntax-table oldsyntax))))

(put 'matlab-navigation-syntax 'lisp-indent-function 0)
(add-hook 'edebug-setup-hook
	  (lambda ()
	    (def-edebug-spec matlab-navigation-syntax def-body)))

(defun matlab-up-list (count &optional restrict)
  "Move forwards or backwards up a list by COUNT.
Optional argument RESTRICT is where to restrict the search."
  ;; Matlab syntax table has no disabling strings or comments.
  (let ((dir (if (> 0 count) -1 +1))
	(origin (point))
	(ms nil))
    ;; Make count positive
    (setq count (* count dir))
    (if (= dir -1)
	(while (/= count 0)
	  ;; Search till we find an unstrung paren object.
	  (setq ms (re-search-backward "\\s(\\|\\s)" restrict t))
	  (while (and (save-match-data (matlab-cursor-in-string-or-comment))
		      (setq ms (re-search-backward "\\s(\\|\\s)" restrict t))))
	  (if (not ms)
	      (progn
		(goto-char origin)
		(error "Scan Error: List missmatch")))
	  ;; View it's match.
	  (let ((s (match-string 0)))
	    (if (string-match "\\s(" s)
		(setq count (1- count))
	      (setq count (1+ count)))))
      (error "Not implemented"))
    ms))

(defun matlab-valid-end-construct-p ()
  "Return non-nil if the end after point terminates a block.
Return nil if it is being used to dereference an array."
  (let ((p (point))
	(err1 t))
    (condition-case nil
	(save-restriction
	  ;; Restrict navigation only to the current command line
	  (save-excursion
	    (matlab-beginning-of-command)
	    (narrow-to-region (point)
			      (progn ;;(matlab-end-of-command (point))
				(end-of-line)
				     (if (> p (point))
					 (progn
					   (setq err1 nil)
					   (error)))
				     (point))))
	  (save-excursion
	    ;; beginning of param list
	    (matlab-up-list -1)
	    ;; backup over the parens.  If that fails
	    (condition-case nil
		(progn
		  (forward-sexp 1)
		  ;; If we get here, the END is inside parens, which is not a
		  ;; valid location for the END keyword.  As such it is being
		  ;; used to dereference array parameters
		  nil)
	      ;; This error means that we have an unterminated paren
	      ;; block, so this end is currently invalid.
	      (error nil))))
      ;; an error means the list navigation failed, which also means we are
      ;; at the top-level
      (error err1))))

;;; Regexps for MATLAB language ===============================================

;; "-pre" means "partial regular expression"
;; "-if" and "-no-if" means "[no] Indent Function"

(defconst matlab-defun-regex "^\\s-*function\\>"
  "Regular expression defining the beginning of a Matlab function.")

(defconst matlab-block-beg-pre-if "function\\|for\\|while\\|if\\|switch\\|try"
  "Keywords which mark the beginning of an indented block.
Includes function.")

(defconst matlab-block-beg-pre-no-if "for\\|while\\|if\\|switch\\|try"
  "Keywords which mark the beginning of an indented block.
Excludes function.")

(defun matlab-block-beg-pre ()
  "Partial regular expression to recognize Matlab block-begin keywords."
  (if matlab-indent-function
      matlab-block-beg-pre-if
    matlab-block-beg-pre-no-if))

(defconst matlab-block-mid-pre
  "elseif\\|else\\|catch"
  "Partial regular expression to recognize Matlab mid-block keywords.")

(defconst matlab-block-end-pre-if
  "end\\(function\\)?\\|function"
  "Partial regular expression to recognize Matlab block-end keywords.")

(defconst matlab-block-end-pre-no-if
  "end"
  "Partial regular expression to recognize Matlab block-end keywords.")

(defun matlab-block-end-pre ()
  "Partial regular expression to recognize Matlab block-end keywords."
  (if matlab-indent-function
      matlab-block-end-pre-if
    matlab-block-end-pre-no-if))

;; Not used.
;;(defconst matlab-other-pre
;;  "function\\|return"
;;  "Partial regular express to recognize Matlab non-block keywords.")

(defconst matlab-endless-blocks
  "case\\|otherwise"
  "Keywords which initialize new blocks, but don't have explicit ends.
Thus, they are endless.  A new case or otherwise will end a previous
endless block, and and end will end this block, plus any outside normal
blocks.")

(defun matlab-block-re ()
  "Regular expression for keywords which begin Matlab blocks."
  (concat "\\(^\\|[;,]\\)\\s-*\\("
 	  (matlab-block-beg-pre) "\\|"
  	  matlab-block-mid-pre "\\|"
 	  (matlab-block-end-pre) "\\|"
 	  matlab-endless-blocks "\\)\\b"))
  
(defun matlab-block-scan-re ()
  "Expression used to scan over matching pairs of begin/ends."
  (concat "\\(^\\|[;,]\\)\\s-*\\("
 	  (matlab-block-beg-pre) "\\|"
 	  (matlab-block-end-pre) "\\)\\b"))

(defun matlab-block-beg-re ()
  "Expression used to find the beginning of a block."
  (concat "\\(" (matlab-block-beg-pre) "\\)"))

(defun matlab-block-mid-re ()
  "Expression used to find block center parts (like else)."
  (concat "\\(" matlab-block-mid-pre "\\)"))

(defun matlab-block-end-re ()
  "Expression used to end a block.  Usually just `end'."
  (concat "\\(" (matlab-block-end-pre) "\\)"))

(defun matlab-block-end-no-function-re ()
  "Expression representing and end if functions are excluded."
  (concat "\\<\\(" matlab-block-end-pre-no-if "\\)\\>"))

(defun matlab-endless-blocks-re ()
  "Expression of block starters that do not have associated ends."
  (concat "\\(" matlab-endless-blocks "\\)"))

(defconst matlab-cline-start-skip "[ \t]*%[ \t]*"
  "*The regular expression for skipping comment start.")

;;; Lists for matlab keywords =================================================

(defvar matlab-keywords-solo
  '("break" "case" "else" "elseif" "end" "for" "function" "if"
    "otherwise" "profile" "switch" "while")
  "Keywords that appear on a line by themselves.")
(defvar matlab-keywords-return
  '("acos" "acosh" "acot" "acoth" "acsch" "asech" "asin" "asinh"
    "atan" "atan2" "atanh" "cos" "cosh" "coth" "csc" "csch" "exp"
    "log" "log10" "log2" "sec" "sech" "sin" "sinh" "tanh"
    "abs" "sign" "sqrt" )
  "List of Matlab keywords that have return arguments.
This list still needs lots of help.")
(defvar matlab-keywords-boolean
  '("all" "any" "exist" "isempty" "isequal" "ishold" "isfinite" "isglobal"
    "isinf" "isletter" "islogical" "isnan" "isprime" "isreal" "isspace"
    "logical")
  "List of keywords that are typically used as boolean expressions.")

(defvar matlab-core-properties
  '("ButtonDownFcn" "Children" "Clipping" "CreateFcn" "DeleteFcn"
    "BusyAction" "HandleVisibility" "HitTest" "Interruptible"
    "Parent" "Selected" "SelectionHighlight" "Tag" "Type"
    "UIContextMenu" "UserData" "Visible")
  "List of properties belonging to all HG objects.")

(defvar matlab-property-lists
  '(("root" .
     ("CallbackObject" "Language" "CurrentFigure" "Diary" "DiaryFile"
      "Echo" "ErrorMessage" "Format" "FormatSpacing" "PointerLocation"
      "PointerWindow" "Profile" "ProfileFile" "ProfileCount"
      "ProfileInterval" "RecursionLimit" "ScreenDepth" "ScreenSize"
      "ShowHiddenHandles" "TerminalHideGraphCommand" "TerminalOneWindow"
      "TerminalDimensions" "TerminalProtocol" "TerminalShowGraphCommand"
      "Units" "AutomaticFileUpdates" ))
    ("axes" .
     ("AmbientLightColor" "Box" "CameraPosition" "CameraPositionMode"
      "CameraTarget" "CameraTargetMode" "CameraUpVector"
      "CameraUpVectorMode" "CameraViewAngle" "CameraViewAngleMode" "CLim"
      "CLimMode" "Color" "CurrentPoint" "ColorOrder" "DataAspectRatio"
      "DataAspectRatioMode" "DrawMode" "FontAngle" "FontName" "FontSize"
      "FontUnits" "FontWeight" "GridLineStyle" "Layer" "LineStyleOrder"
      "LineWidth" "NextPlot" "PlotBoxAspectRatio" "PlotBoxAspectRatioMode"
      "Projection" "Position" "TickLength" "TickDir" "TickDirMode" "Title"
      "Units" "View" "XColor" "XDir" "XGrid" "XLabel" "XAxisLocation" "XLim"
      "XLimMode" "XScale" "XTick" "XTickLabel" "XTickLabelMode" "XTickMode"
      "YColor" "YDir" "YGrid" "YLabel" "YAxisLocation" "YLim" "YLimMode"
      "YScale" "YTick" "YTickLabel" "YTickLabelMode" "YTickMode" "ZColor"
      "ZDir" "ZGrid" "ZLabel" "ZLim" "ZLimMode" "ZScale" "ZTick"
      "ZTickLabel" "ZTickLabelMode" "ZTickMode"))
    ("figure" .
     ("BackingStore" "CloseRequestFcn" "Color" "Colormap"
      "CurrentAxes" "CurrentCharacter" "CurrentObject" "CurrentPoint"
      "Dithermap" "DithermapMode" "FixedColors" "IntegerHandle"
      "InvertHardcopy" "KeyPressFcn" "MenuBar" "MinColormap" "Name"
      "NextPlot" "NumberTitle" "PaperUnits" "PaperOrientation"
      "PaperPosition" "PaperPositionMode" "PaperSize" "PaperType"
      "Pointer" "PointerShapeCData" "PointerShapeHotSpot" "Position"
      "Renderer" "RendererMode" "Resize" "ResizeFcn" "SelectionType"
      "ShareColors" "Units" "WindowButtonDownFcn"
      "WindowButtonMotionFcn" "WindowButtonUpFcn" "WindowStyle"))
    ("image" . ("CData" "CDataMapping" "EraseMode" "XData" "YData"))
    ("light" . ("Position" "Color" "Style"))
    ("line" .
     ("Color" "EraseMode" "LineStyle" "LineWidth" "Marker"
      "MarkerSize" "MarkerEdgeColor" "MarkerFaceColor" "XData" "YData"
      "ZData"))
    ("patch" .
     ("CData" "CDataMapping" "FaceVertexCData" "EdgeColor" "EraseMode"
      "FaceColor" "Faces" "LineStyle" "LineWidth" "Marker"
      "MarkerEdgeColor" "MarkerFaceColor" "MarkerSize" "Vertices"
      "XData" "YData" "ZData" "FaceLighting" "EdgeLighting"
      "BackFaceLighting" "AmbientStrength" "DiffuseStrength"
      "SpecularStrength" "SpecularExponent" "SpecularColorReflectance"
      "VertexNormals" "NormalMode"))
    ("surface" .
     ("CData" "CDataMapping" "EdgeColor" "EraseMode" "FaceColor"
      "LineStyle" "LineWidth" "Marker" "MarkerEdgeColor"
      "MarkerFaceColor" "MarkerSize" "MeshStyle" "XData" "YData"
      "ZData" "FaceLighting" "EdgeLighting" "BackFaceLighting"
      "AmbientStrength" "DiffuseStrength" "SpecularStrength"
      "SpecularExponent" "SpecularColorReflectance" "VertexNormals"
      "NormalMode"))
    ("text\\|title\\|xlabel\\|ylabel\\|zlabel" .
     ("Color" "EraseMode" "Editing" "Extent" "FontAngle" "FontName"
      "FontSize" "FontUnits" "FontWeight" "HorizontalAlignment"
      "Position" "Rotation" "String" "Units" "Interpreter"
      "VerticalAlignment"))
    ("uicontextmenu" . ("Callback"))
    ("uicontrol" .
     ("BackgroundColor" "Callback" "CData" "Enable" "Extent"
      "FontAngle" "FontName" "FontSize" "FontUnits" "FontWeight"
      "ForegroundColor" "HorizontalAlignment" "ListboxTop" "Max" "Min"
      "Position" "String" "Style" "SliderStep" "TooltipString" "Units"
      "Value"))
    ("uimenu" .
     ("Accelerator" "Callback" "Checked" "Enable" "ForegroundColor"
      "Label" "Position" "Separator"))
    ;; Flesh this out more later.
    ("uipushtool\\|uitoggletool\\|uitoolbar" .
     ("Cdata" "Callback" "Separator" "Visible"))
    )
  "List of property lists on a per object type basis.")

(defvar matlab-unknown-type-commands
  "[gs]et\\|findobj\\|waitfor"
  "Expression for commands that have unknown types.")

(defun matlab-all-known-properties ()
  "Return a list of all properties."
  (let ((lst matlab-core-properties)
	(tl matlab-property-lists))
    (while tl
      (setq lst (append lst (cdr (car tl)))
	    tl (cdr tl)))
    (matlab-uniquafy-list lst)))

(defvar matlab-all-known-properties (matlab-all-known-properties)
  "List of all the known properties.")

(defmacro matlab-property-function ()
  "Regexp of all builtin functions that take property lists."
  '(let ((r matlab-unknown-type-commands)
	 (tl matlab-property-lists))
     (while tl
       (setq r (concat r "\\|" (car (car tl)))
	     tl (cdr tl)))
     r))

;;; Navigation ===============================================================

(defvar matlab-scan-on-screen-only nil
  "When this is set to non-nil, then forward/backward sexp stops off screen.
This is so the block highlighter doesn't gobble up lots of time when
a block is not terminated.")

(defun matlab-backward-sexp (&optional autoend noerror)
  "Go backwards one balanced set of Matlab expressions.
If optional AUTOEND, then pretend we are at an end.
If optional NOERROR, then we return t on success, and nil on failure."
  (interactive "P")
  (matlab-navigation-syntax
    (if (and (not autoend)
	     (save-excursion (backward-word 1)
			     (or (not
				  (and (looking-at
					(matlab-block-end-no-function-re))
				       (matlab-valid-end-construct-p)))
				 (matlab-cursor-in-string-or-comment))))
	;; Go backwards one simple expression
	(forward-sexp -1)
      ;; otherwise go backwards recursively across balanced expressions
      ;; backup over our end
      (if (not autoend) (forward-word -1))
      (let ((done nil) (start (point)) (returnme t))
	(while (and (not done)
		    (or (not matlab-scan-on-screen-only)
			(pos-visible-in-window-p)))
	  (if (re-search-backward (matlab-block-scan-re) nil t)
	      (progn
		(goto-char (match-beginning 2))
		(if (looking-at (matlab-block-end-no-function-re))
		    (if (or (matlab-cursor-in-string-or-comment)
			    (not (matlab-valid-end-construct-p)))
			nil
		      ;; we must skip the expression and keep searching
		      (forward-word 1)
		      (matlab-backward-sexp))
		  (if (not (matlab-cursor-in-string-or-comment))
		      (setq done t))))
	    (goto-char start)
	    (if noerror
		(setq returnme nil)
	      (error "Unstarted END construct"))))
	returnme))))
  
(defun matlab-forward-sexp (&optional includeelse)
    "Go forward one balanced set of Matlab expressions.
Optional argument INCLUDEELSE will stop on ELSE if it matches the starting IF."
  (interactive "P")
  (matlab-navigation-syntax
    ;; skip over preceeding whitespace
    (skip-chars-forward " \t\n;")
    (if (or (not (looking-at (concat "\\("
				     (matlab-block-beg-pre)
				     "\\|"
				     (matlab-block-mid-re)
				     "\\)\\>")))
	    (matlab-cursor-in-string-or-comment))
	;; Go forwards one simple expression
	(forward-sexp 1)
      ;; otherwise go forwards recursively across balanced expressions
      (forward-word 1)
      (let ((done nil) (s nil)
	    (expr-scan (if includeelse
			   (matlab-block-re)
			 (matlab-block-scan-re)))
	    (expr-look (matlab-block-beg-pre)))
	(while (and (not done)
		    (setq s (re-search-forward expr-scan nil t))
		    (or (not matlab-scan-on-screen-only)
			(pos-visible-in-window-p)))
	  (goto-char (match-beginning 2))
	  (if (looking-at expr-look)
	      (if (matlab-cursor-in-string-or-comment)
		  (forward-word 1)
		;; we must skip the expression and keep searching
		;; NEVER EVER call with value of INCLUDEELSE
		(matlab-forward-sexp))
	    (forward-word 1)
	    (if (and (not (matlab-cursor-in-string-or-comment))
		     (matlab-valid-end-construct-p))
		(setq done t))))
	(if (not s) (error "Unterminated block"))))))

(defun matlab-beginning-of-defun ()
  "Go to the beginning of the current function."
  (interactive)
  (or (re-search-backward matlab-defun-regex nil t)
      (goto-char (point-min))))

(defun matlab-end-of-defun ()
  "Go to the end of the current function."
  (interactive)
  (or (progn
	(if (looking-at matlab-defun-regex) (goto-char (match-end 0)))
	(if (re-search-forward matlab-defun-regex nil t)
	    (progn (forward-line -1)
		   t)))
      (goto-char (point-max))))

(defun matlab-beginning-of-command ()
  "Go to the beginning of an M command.
Travels across continuations."
  (interactive)
  (beginning-of-line)
  (let ((p nil)
	;; This restriction is a wild guess where to end reverse
	;; searching for array continuations.  The reason is that
	;; matlab up list is very slow, and most people would never
	;; put a blank line in a matrix.  Either way, it's worth the
	;; trade off to speed this up for large files.
	;; This list of keywords is NOT meant to be comprehensive.
	(r (save-excursion
	     (re-search-backward
	      "^\\s-*\\(%\\|if\\|else\\(if\\)\\|while\\|for\\|$\\)\\>"
	      nil t))))
    (while (and (or (save-excursion (and (matlab-prev-line)
					 (matlab-lattr-cont)))
		    (setq p (matlab-lattr-array-cont r)))
		(save-excursion (beginning-of-line) (not (bobp))))
      (if p (goto-char p) (matlab-prev-line))
      (setq p nil))
    (back-to-indentation)))

(defun matlab-end-of-command (&optional beginning)
  "Go to the end of an M command.
Optional BEGINNING is where the command starts from."
  (interactive)
  (while (and (or (matlab-lattr-cont)
		  (save-excursion
		    (forward-line 1)
		    (and (not (eobp))
			 (or (matlab-ltype-continued-comm)
			     (matlab-lattr-array-cont beginning)))))
	      ;; This hack is a short circuit.  If a user did not
	      ;; correctly end a matrix, this will short-circuit
	      ;; as soon as somethng that would never appear in a matrix
	      ;; becomes visible.
	      (not (save-excursion
		     (beginning-of-line)
		     (looking-at (matlab-block-scan-re)))))
    (forward-line 1))
  (end-of-line))


;;; Line types and attributes =================================================

(defun matlab-ltype-empty ()		; blank line
  "Return t if current line is empty."
  (save-excursion
    (beginning-of-line)
    (looking-at "^[ \t]*$")))

(defun matlab-ltype-comm ()		; comment line
  "Return t if current line is a MATLAB comment line."
  (save-excursion
    (beginning-of-line)
    (looking-at "[ \t]*%.*$")))

(defun matlab-ltype-comm-ignore ()	; comment out a region line
  "Return t if current line is a MATLAB comment region line."
  (save-excursion
    (beginning-of-line)
    (looking-at (concat "[ \t]*" matlab-comment-region-s))))

(defun matlab-ltype-help-comm ()
  "Return t if the current line is part of the MATLAB help comment."
  (save-excursion
    (if (not (matlab-ltype-comm))
	nil
      (while (and (matlab-ltype-comm) (not (bobp))
		  (matlab-prev-line))
	(beginning-of-line))
      (matlab-ltype-function-definition))))

(defun matlab-ltype-endfunction-comm ()
  "Return t if the current line is an ENDFUNCTION style comment."
  (save-excursion
    (if (not (matlab-ltype-comm))
	nil
      (beginning-of-line)
      (if (looking-at "^[ \t]*%[ \t]*endfunction")
	  t
	(while (and (or (matlab-ltype-comm)
			(matlab-ltype-empty))
		    (not (eobp)))
	  (forward-line 1))
	(matlab-ltype-function-definition)))))

(defun matlab-ltype-continued-comm ()
  "Return column of previous line's comment start, or nil."
  (save-excursion
    (beginning-of-line)
    (if (or (not (matlab-ltype-comm)) (bobp))
	nil
      ;; We use forward-line and not matlab-prev-line because
      ;; we want blank lines to terminate this indentation method.
      (forward-line -1)
      (let ((col  (matlab-lattr-comm)))
	(if col
	    (progn
	      (goto-char col)
	      (current-column))
	  nil)))))

(defun matlab-ltype-function-definition ()
  "Return t if the current line is a function definition."
  (save-excursion
    (beginning-of-line)
    (looking-at matlab-defun-regex)))

(defun matlab-ltype-code ()		; line of code
  "Return t if current line is a MATLAB code line."
  (and (not (matlab-ltype-empty)) (not (matlab-ltype-comm))))

(defun matlab-lattr-comm ()		; line has comment
  "Return t if current line contain a comment."
  (save-excursion (matlab-comment-on-line)))

(defun matlab-lattr-cont ()		; line has continuation
  "Return non-nil if current line ends in ... and optional comment."
  (save-excursion
    (beginning-of-line)
    (and (re-search-forward "[^ \t.][ \t]*\\.\\.+[ \t]*\\(%.*\\)?$"
			    (matlab-point-at-eol) t)
	 (progn (goto-char (match-beginning 0))
		(not (matlab-cursor-in-comment))))))

(defun matlab-lattr-array-cont (&optional restrict)
  "Return non-nil if current line is in an array.
If the entirety of the array is on this line, return nil.
Optional option RESTRICT is the distrance to restrict the search."
  (condition-case nil
      (save-excursion
	(beginning-of-line)
	(matlab-up-list -1 restrict)
	(and (looking-at "[[{]") (point)))
    (error nil)))

(defun matlab-lattr-array-end ()
  "Return non-nil if the current line closes an array.
by close, the first character is the end of an array."
  (save-excursion
    (back-to-indentation)
    (and (looking-at "[]}]") (matlab-lattr-array-cont))))

(defun matlab-lattr-block-cont (&optional eol)
  "Return a number representing the number of unterminated block constructs.
This is any block, such as if, or for that doesn't have an END on this line.
Optional EOL indicates a virtual end of line."
  (let ((v 0))
    (save-excursion
      (beginning-of-line)
      (save-restriction
	(narrow-to-region (point) (or eol (matlab-point-at-eol)))
	(matlab-navigation-syntax
	  (while (re-search-forward (concat "\\<" (matlab-block-beg-re) "\\>")
				    nil t)
	    (if (matlab-cursor-in-string-or-comment)
		;; Do nothing
		nil
	      ;; Increment counter, move to end.
	      (setq v (1+ v))
	      (let ((p (point)))
		(forward-word -1)
		(condition-case nil
		    (progn
		      (matlab-forward-sexp)
		      (setq v (1- v)))
		  (error (goto-char p))))))
	  (if (= v 0) nil v))))))

(defun matlab-lattr-middle-block-cont ()
  "Return the number of middle block continuations.
This should be 1 or nil, and only true if the line starts with one of these
special items."
  (save-excursion
    (back-to-indentation)
    (if (looking-at (concat (matlab-block-mid-re) "\\>"))
	(if (and (re-search-forward (matlab-block-end-pre)
				    (matlab-point-at-eol)
				    t)
		 (matlab-valid-end-construct-p))
	    ;; If there is an END, we still need to return non-nil,
	    ;; but the number value is a net of 0.
	    0
	  1)
      nil)))

(defun matlab-lattr-endless-block-cont ()
  "Return the number of middle block continuations.
This should be 1 or nil, and only true if the line starts with one of these
special items."
  (save-excursion
    (back-to-indentation)
    (if (looking-at (concat (matlab-endless-blocks-re) "\\>"))
	1
      nil)))

(defun matlab-lattr-block-close ()
  "Return the number of closing block constructs. (not used yet)."
  (let ((v 0))
    (save-excursion
      (end-of-line)
      (save-restriction
	(narrow-to-region (matlab-point-at-bol) (point))
	(while (and (re-search-backward (matlab-block-end-re) nil t)
		    (matlab-valid-end-construct-p))
	  (setq v (1+ v))
	  (condition-case nil
	      (progn
		(matlab-backward-sexp t)
		(setq v (1- v)))
	    (error nil)))
	(if (= v 0) nil v)))))

(defun matlab-lattr-local-end ()
  "Return t if this line begins with an end construct."
  (save-excursion
    (back-to-indentation)
    (let ((matlab-indent-function nil))
      (and (looking-at (concat "\\<" (matlab-block-end-re) "\\>"))
	   (matlab-valid-end-construct-p)))))

(defun matlab-lattr-semantics (&optional prefix)
  "Return the semantics of the current position.
Values are nil 'solo, 'value, and 'boolean.  Boolean is a subset of
value.  nil means there is no semantic content (ie, string or comment.)
If optional PREFIX, then return 'solo if that is the only thing on the
line."
  (cond ;((matlab-cursor-in-string-or-comment)
	 ;nil)
	((or (matlab-ltype-empty)
	     (and prefix (save-excursion
			   (beginning-of-line)
			   (looking-at (concat "\\s-*" prefix "\\s-*$")))))
	 'solo)
	((save-excursion
	   (matlab-beginning-of-command)
	   (looking-at "\\s-*\\(if\\|elseif\\|while\\)\\>"))
	 'boolean)
	((save-excursion
	   (matlab-beginning-of-command)
	   (looking-at (concat "\\s-*\\(" (matlab-property-function)
			       "\\)\\>")))
	 'property)
	(t
	 'value)))

(defun matlab-function-called-at-point ()
  "Return a string representing the function called nearby point."
  (save-excursion
    (beginning-of-line)
    (cond ((looking-at "\\s-*\\([a-zA-Z]\\w+\\)[^=][^=]")
	   (match-string 1))
	  ((and (re-search-forward "=" (matlab-point-at-eol) t)
		(looking-at "\\s-*\\([a-zA-Z]\\w+\\)\\s-*[^=]"))
	   (match-string 1))
	  (t nil))))

(defun matlab-cursor-in-string-or-comment ()
  "Return t if the cursor is in a valid Matlab comment or string."
  ;; comment and string depend on each other.  Here is one test
  ;; that does both.
  (save-restriction
    (narrow-to-region (matlab-point-at-bol) (matlab-point-at-eol))
    (let ((p (1+ (point)))
	  (returnme nil)
	  (sregex (concat matlab-string-start-regexp "'")))
      (save-excursion
	(goto-char (point-min))
	(while (and (re-search-forward
		     (concat "'\\|%\\|" (regexp-quote matlab-elipsis-string))
		     nil t)
		    (<= (point) p))
	  (if (or (= ?% (preceding-char))
		  (= ?. (preceding-char)))
	      ;; Here we are in a comment for the rest of it.
	      (progn
		(goto-char p)
		(setq returnme t))
	    ;; Here, we could be a string start, or transpose...
	    (if (or (= (current-column) 1)
		    (save-excursion (forward-char -2)
				    (looking-at sregex)))
		;; a valid string start, find the end
		(let ((f (re-search-forward matlab-string-end-regexp nil t)))
		  (if f
		      (setq returnme (> (point) p))
		    (setq returnme t)))
	      ;; Ooops, a transpose, keep going.
	      ))))
      returnme)))

(defun matlab-cursor-in-comment ()
  "Return t if the cursor is in a valid Matlab comment."
  (save-match-data
    (save-restriction
      (narrow-to-region (matlab-point-at-bol) (matlab-point-at-eol))
      (save-excursion
	(let ((prev-match nil))
	  (while (and (re-search-backward
		       (concat "%\\|" (regexp-quote matlab-elipsis-string) "+")
		       nil t)
		      (not (matlab-cursor-in-string)))
	    (setq prev-match (point)))
	  (if (and prev-match (matlab-cursor-in-string))
	      (goto-char prev-match))
	  (and (looking-at (concat "%\\|"
				   (regexp-quote matlab-elipsis-string)))
	       (not (matlab-cursor-in-string))))))))

(defun matlab-cursor-in-string (&optional incomplete)
  "Return t if the cursor is in a valid Matlab string.
If the optional argument INCOMPLETE is non-nil, then return t if we
are in what could be a an incomplete string."
  (let ((m (match-data))
	(returnme nil))
    (save-restriction
      (narrow-to-region (matlab-point-at-bol) (matlab-point-at-eol))
      (let ((p (1+ (point)))

	    (sregex (concat matlab-string-start-regexp "'"))
	    (instring nil))
	(save-excursion
	  ;; Comment hunters need strings to not call the comment
	  ;; identifiers.  Thus, this routines must be savvy of comments
	  ;; without recursing to them.
	  (goto-char (point-min))
	  (while (or (and instring (looking-at "'"))
		     (and (re-search-forward
			   (concat "'\\|%\\|"
				   (regexp-quote matlab-elipsis-string))
			   nil t)
			  (<= (point) p)
			  ;; Short circuit to fix this.
			  (progn (setq instring nil) t)))
	    ;; The next line emulates re-search-foward
	    (if instring (goto-char (match-end 0)))
	    (if (or (= ?% (preceding-char))
		    (= ?. (preceding-char)))
		;; Here we are in a comment for the rest of it.
		;; thus returnme is a force-false.
		(goto-char p)
	      ;; Here, we could be in a string start, or transpose...
	      (if (or (= (current-column) 1)
		      instring
		      (save-excursion (forward-char -2)
				      (looking-at sregex)))
		  ;; a valid string start, find the end
		  (let ((f (re-search-forward matlab-string-end-regexp nil t)))
		    (if (and (not f) incomplete)
			(setq returnme t)
		      (setq returnme (> (point) p))
		      (setq instring t)))
		;; Ooops, a transpose, keep going.
		))))))
    (set-match-data m)
    returnme))
  

(defun matlab-comment-on-line ()
  "Place the cursor on the beginning of a valid comment on this line.
If there isn't one, then return nil, point otherwise."
  (interactive)
  (let ((eol (matlab-point-at-eol))
	(p (point))
	(signal-error-on-buffer-boundary nil))
    (beginning-of-line)
    (while (and (re-search-forward "%" eol t)
		(save-excursion (forward-char -1) (matlab-cursor-in-string t))))
    (if (not (bolp)) (forward-char -1))
    (if (looking-at "%")
	(point)
      (goto-char p)
      nil)))

;;; Indent functions ==========================================================

(defun matlab-indent-line ()
  "Indent a line in `matlab-mode'."
  (interactive)
  (let ((i (matlab-calc-indent))
	(c (current-column)))
    (save-excursion
      (back-to-indentation)
      (if (= i (current-column))
	  nil
	(beginning-of-line)
	(delete-horizontal-space)
	(indent-to i))
      ;; If line contains a comment, format it.
      (if () (if (matlab-lattr-comm) (matlab-comment))))
    (if (<= c i) (move-to-column i))))

(defun matlab-calc-indent ()
  "Return the appropriate indentation for this line as an integer."
  (interactive)
  (let* ((tli (save-excursion (matlab-beginning-of-command)
			      (matlab-point-at-bol)))
	 (ci (save-excursion (or (matlab-prev-line)
				 (progn (beginning-of-line)
					(forward-line -1)))
			     (matlab-beginning-of-command)
			     (matlab-next-line-indentation)))
	 (sem (matlab-calculate-indentation (if (>= ci tli) 0 ci))))
    ;; simplistic
    (nth 1 sem)))

(defun matlab-calculate-indentation (current-indentation)
  "Calculate out the indentation of the current line.
Return a list of descriptions for this line.  Return format is:
 '(TYPE DEPTHNUMBER)
where TYPE is one of (comment, code, function, blockstart, blockmid,
blockendless, blockend) DEPTHNUMBER is how many characters to indent
this line.
  Argument CURRENT-INDENTATION is what the previous line thinks
this line's indentation should be."
  (let ((ci current-indentation)
	(tmp nil))
    (cond
     ;; COMMENTS
     ((matlab-ltype-comm)
      (cond
       ;; HELP COMMENT and COMMENT REGION
       ((or (matlab-ltype-help-comm)
	    (matlab-ltype-comm-ignore))
	(list 'comment-help 0))
       ;; COMMENT Continued From Previous Line
       ((setq tmp (matlab-ltype-continued-comm))
	(list 'comment tmp))
       ;; END FUNCTION COMMENT
       ((matlab-ltype-endfunction-comm)
	(list 'comment-endfunction 0))
       (t
	(list 'comment ci))))
     ;; FUNCTION DEFINITION
     ((matlab-ltype-function-definition)
      (list 'function 0))
     ;; END keyword
     ((matlab-lattr-local-end)
      (list 'blockend (save-excursion
			(beginning-of-line)
			(condition-case nil
			    (progn
			      (matlab-backward-sexp t)
			      (if (matlab-ltype-function-definition) (error ""))
			      (forward-word 1) ;; skip this match when counting
			      (+ (current-indentation)
				 (* (1- (matlab-lattr-block-cont (point)))
				    matlab-indent-level)))
			  (error (error "Unmatched end"))))))
     ;; ELSE/CATCH keywords
     ((matlab-lattr-middle-block-cont)
      (let ((m (match-string 1)))
	(list 'blockmid
	      (condition-case nil
		  (save-excursion
		    (beginning-of-line)
		    (matlab-backward-sexp t)
		    (if (matlab-ltype-function-definition) (error ""))
		    (current-column))
		(error (error "Unmatched %s" m))))))
     ;; CASE/OTHERWISE keywords
     ((matlab-lattr-endless-block-cont)
      (list 'blockendless
	    (condition-case nil
		(save-excursion
		  (beginning-of-line)
		  (matlab-backward-sexp t)
		  (if (not (looking-at "switch\\>")) (error ""))
		  (+ (current-column)
		     (if (listp matlab-case-level)
			 (car matlab-case-level)
		       matlab-case-level)))
	      (error (error "Unmatched case/otherwise part")))))
     ;; End of a MATRIX
     ((matlab-lattr-array-end)
      (list 'array-end (save-excursion
			(back-to-indentation)
			(matlab-up-list -1)
			(let* ((fc (following-char))
			       (mi (assoc fc matlab-maximum-indents))
			       (max (if mi (if (listp (cdr mi))
					       (car (cdr mi)) (cdr mi))
				      nil))
			       (ind (if mi (if (listp (cdr mi))
					       (cdr (cdr mi)) (cdr mi))
				      nil)))
			  ;; apply the maximum limits.
			  (if (and ind (> (- (current-column) ci) max))
			      (1- ind) ; decor
			    (current-column))))))
     ;; Code lines
     ((save-excursion
	(beginning-of-line)
	(back-to-indentation)
	(= (point) (progn (matlab-beginning-of-command) (point))))
      ;; This means we are at the beginning of a command structure.
      ;; Always match up against the previous line.
      (list 'code ci))
     ;; Lines continued from previous statements.
     (t
      (list (if (matlab-ltype-empty) 'empty
	      (if (matlab-lattr-array-cont) 'array-cont 'code))
	    (condition-case nil
		;; Line up with opening paren/brace/bracket
		(let ((boc (save-excursion
			     (matlab-beginning-of-command)
			     (point))))
		  (save-excursion
		    (beginning-of-line)
		    (matlab-up-list -1)
		    (if (> boc (point)) (error nil))
		    ;; Ok, it MIGHT be that we are in a program
		    ;; statement, and this particular command is an HG
		    ;; statement that would look better if the
		    ;; following lines lined up AFTER the first
		    ;; argument.  Lets look.
		    (let ((parendepth (current-column)))
		      (cond ((and (= (following-char) ?\( )
				  (save-excursion
				    (matlab-navigation-syntax
				      (forward-word -1)
				      (looking-at
				       matlab-indent-past-arg1-functions)))
				  (let ((start-paren (point)))
				    (while
					(and
					 (re-search-forward
					  "," (matlab-point-at-eol) t)
					 (save-excursion
					   (matlab-up-list -1)
					   (> (point) start-paren))))
				    (if (and
					 (= (preceding-char) ?,)
					 ;; Don't bother if we hit the EOL.
					 (not (looking-at

					       "\\s-*\\(\\.\\.\\.\\|$\\|)\\)")))
					t
				      (move-to-column parendepth)
				      nil)))
			     (skip-chars-forward " \t")
			     (if (> (- (current-column) parendepth)
				    matlab-arg1-max-indent-length)
				 (+ parendepth matlab-arg1-max-indent-length)
			       (current-column)))
			    (t
			     (let* ((fc (following-char))
				    (mi (assoc fc matlab-maximum-indents))
				    (max (if mi
					     (if (listp (cdr mi))
						 (car (cdr mi)) (cdr mi))
					   nil))
				    (ind (if mi
					     (if (listp (cdr mi))
						 (cdr (cdr mi)) (cdr mi))
					   nil)))
			       (forward-char 1)
			       (skip-chars-forward " \t")
			       ;; If we are at the end of a line and
			       ;; this open paren is there, then we
			       ;; DONT want to indent to it.  Use the
			       ;; standard indent.
			       (if (looking-at "\\.\\.\\.\\|$")
				   ;; This could happen in another set
				   ;; of matricies.  Find a current
				   ;; indentation based on the
				   ;; previous line.
				   (let ((cci (current-indentation)))
				     (+ cci matlab-cont-level))
				 ;; apply the maximum limits.
				 (if (and ind (> (- (current-column) ci) max))
				     ind
				   (current-column)))))))))
	      (error
	       ;; Line up to an equals sign.
	       (save-excursion
		 (matlab-beginning-of-command)
		 (while (and (re-search-forward "=" (matlab-point-at-eol) t)
			     (matlab-cursor-in-string-or-comment)))
		 (if (/= (preceding-char) ?=)
		     (+ ci matlab-cont-level)
		   (skip-chars-forward " \t")
		   (let ((cc (current-column))
			 (mi (assoc ?= matlab-maximum-indents)))
		     (if (looking-at "\\.\\.\\.\\|$")
			 ;; In this case, the user obviously wants the
			 ;; indentation to be somewhere else.
			 (+ ci (cdr (cdr mi)))
		       ;; If the indent delta is greater than the max,
		       ;; use the max + currenti
		       (if (and mi (> (- cc ci) (if (listp (cdr mi))
						    (car (cdr mi))
						  (cdr mi))))
			   (setq cc (+ ci (if (listp (cdr mi))
					      (cdr (cdr mi))
					    (cdr mi)))))
		       cc))))))))
     )))

(defun matlab-next-line-indentation ()
  "Calculate the indentation for lines preceeding this command line."
  (let ((bc (matlab-lattr-block-cont))
	(mc (matlab-lattr-middle-block-cont))
	(ec (matlab-lattr-endless-block-cont))
	(hc (and matlab-indent-function (matlab-ltype-help-comm))))
    (+ (current-indentation)
       (* matlab-indent-level (or bc 0))
       (* matlab-indent-level (or mc 0))
       (* (if (listp matlab-case-level)
	      (cdr matlab-case-level) matlab-case-level)
	  (or ec 0))
       (if hc matlab-indent-level 0))))

;;; The return key ============================================================

(defcustom matlab-return-function 'matlab-indent-end-before-ret
  "Function to handle return key.
Must be one of:
    'matlab-plain-ret
    'matlab-indent-after-ret
    'matlab-indent-end-before-ret
    'matlab-indent-before-ret"
  :group 'matlab
  :type '(choice (function-item matlab-plain-ret)
		 (function-item matlab-indent-after-ret)
		 (function-item matlab-indent-end-before-ret)
		 (function-item matlab-indent-before-ret)))

(defun matlab-return ()
  "Handle carriage return in `matlab-mode'."
  (interactive)
  (funcall matlab-return-function))

(defun matlab-plain-ret ()
  "Vanilla new line."
  (interactive)
  (newline))
  
(defun matlab-indent-after-ret ()
  "Indent after new line."
  (interactive)
  (newline)
  (matlab-indent-line))

(defun matlab-indent-end-before-ret ()
  "Indent line if block end, start new line, and indent again."
  (interactive)
  (if (save-excursion
	(beginning-of-line)
	(looking-at (concat "^\\s-*\\(" (matlab-block-end-re)
			    "\\|" (matlab-block-mid-re)
			    "\\|" (matlab-endless-blocks-re)
			    "\\)")))
      (condition-case nil
	  (matlab-indent-line)
	(error nil)))
  (newline)
  (matlab-indent-line))

(defun matlab-indent-before-ret ()
  "Indent line, start new line, and indent again."
  (interactive)
  (matlab-indent-line)
  (newline)
  (matlab-indent-line))

(defun matlab-linefeed ()
  "Handle line feed in `matlab-mode'.
Has effect of `matlab-return' with (not matlab-indent-before-return)."
  (interactive)
  (matlab-indent-line)
  (newline)
  (matlab-indent-line))

(defun matlab-comment-return ()
  "Handle carriage return for Matlab comment line."
  (interactive)
  (cond
   ((matlab-ltype-comm)
    (matlab-set-comm-fill-prefix) (newline) (insert fill-prefix)
    (matlab-reset-fill-prefix) (matlab-indent-line))
   ((matlab-lattr-comm)
    (newline) (indent-to comment-column)
    (insert matlab-comment-on-line-s))
   (t
    (newline) (matlab-comment) (matlab-indent-line))))

(defun matlab-comm-from-prev ()
  "If the previous line is a comment-line then set up a comment on this line."
  (save-excursion
    ;; If the previous line is a comment-line then set the fill prefix from
    ;; the previous line and fill this line.
    (if (and (= 0 (forward-line -1)) (matlab-ltype-comm))
	(progn
	  (matlab-set-comm-fill-prefix)
	  (forward-line 1) (beginning-of-line)
	  (delete-horizontal-space)
	  (if (looking-at "%") (delete-char 1))
	  (delete-horizontal-space)
	  (insert fill-prefix)
	  (matlab-reset-fill-prefix)))))

;;; Comment management========================================================

(defun matlab-comment ()
  "Add a comment to the current line."
  (interactive)
  (cond ((matlab-ltype-empty)		; empty line
	 (matlab-comm-from-prev)
	 (if (matlab-lattr-comm)
	     (skip-chars-forward " \t%")
	   (insert matlab-comment-line-s)
	   (matlab-indent-line)))
	((matlab-ltype-comm)		; comment line
	 (matlab-comm-from-prev)
	 (skip-chars-forward " \t%"))
	((matlab-lattr-comm)		; code line w/ comment
	 (beginning-of-line)
	 (re-search-forward "[^%]%[ \t]")
	 (forward-char -2)
	 (if (> (current-column) comment-column) (delete-horizontal-space))
	 (if (< (current-column) comment-column) (indent-to comment-column))
	 (skip-chars-forward "% \t"))
	(t				; code line w/o comment
	 (end-of-line)
	 (re-search-backward "[^ \t\n^]" 0 t)
	 (forward-char)
	 (delete-horizontal-space)
	 (if (< (current-column) comment-column)
	     (indent-to comment-column)
	   (insert " "))
	 (insert matlab-comment-on-line-s))))

(defun matlab-comment-line-break-function (&optional soft)
  "Break the current line, and if in a comment, continue it.
Optional argument SOFT indicates that the newline is soft, and not hard."
  (interactive)
  (if (not (matlab-cursor-in-comment))
      (matlab-return)
    ;; Will the below fn work in old emacsen?
    (if soft (insert-and-inherit ?\n) (newline 1))
    (insert "% ")
    (matlab-indent-line)
    (end-of-line)))

(defun matlab-comment-indent ()
  "Indent a comment line in `matlab-mode'."
  (matlab-calc-indent))

(defun matlab-comment-region (beg-region end-region arg)
  "Comments every line in the region.
Puts `matlab-comment-region-s' at the beginning of every line in the region.
BEG-REGION and END-REGION are arguments which specify the region boundaries.
With non-nil ARG, uncomments the region."
  (interactive "*r\nP")
  (let ((end-region-mark (make-marker)) (save-point (point-marker)))
    (set-marker end-region-mark end-region)
    (goto-char beg-region)
    (beginning-of-line)
    (if (not arg)			;comment the region
	(progn (insert matlab-comment-region-s)
	       (while (and  (= (forward-line 1) 0)
			    (< (point) end-region-mark))
		 (insert matlab-comment-region-s)))
      (let ((com (regexp-quote matlab-comment-region-s))) ;uncomment the region
	(if (looking-at com)
	    (delete-region (point) (match-end 0)))
	(while (and  (= (forward-line 1) 0)
		     (< (point) end-region-mark))
	  (if (looking-at com)
	      (delete-region (point) (match-end 0))))))
    (goto-char save-point)
    (set-marker end-region-mark nil)
    (set-marker save-point nil)))

(defun matlab-uncomment-region (beg end)
  "Uncomment the current region if it is commented out.
Argument BEG and END indicate the region to uncomment."
  (interactive "*r")
  (matlab-comment-region beg end t))

;;; Filling ===================================================================

(defun matlab-set-comm-fill-prefix ()
  "Set the `fill-prefix' for the current (comment) line."
  (interactive)
  (if (matlab-lattr-comm)
      (setq fill-prefix
	    (save-excursion
	      (beginning-of-line)
	      (let ((e (matlab-point-at-eol))
		    (pf nil))
		(while (and (re-search-forward "%[ \t]*\\($$$ \\)?" e t)
			    (matlab-cursor-in-string)))
		(setq pf (match-string 0))
		(concat (make-string (- (current-column) (length pf)) ? )
			pf))))))

(defun matlab-set-comm-fill-prefix-post-code ()
  "Set the `fill-prefix' for the current post-code comment line."
  (interactive)
  (matlab-set-comm-fill-prefix))

(defun matlab-reset-fill-prefix ()
  "Reset the `fill-prefix'."
  (setq fill-prefix nil))

(defun matlab-find-convenient-line-break ()
  "For the current line, position the cursor where we want to break the line.
Basically, spaces are best, then operators.  Always less than `fill-column'
unless we decide we can fudge the numbers.  Return nil if this line should
not be broken.  This function will ONLY work on code."
  ;; First of all, if this is a continuation, then the user is
  ;; requesting that we don't mess with his stuff.
  (if (matlab-lattr-cont)
      nil
    (save-restriction
      (narrow-to-region (matlab-point-at-bol) (matlab-point-at-eol))
      ;; get ourselves onto the fill-column.
      (move-to-column fill-column)
      (let ((pos nil)
	    (orig (point)))
	(or
	 ;; Next, if we have a trailing comment, use that.
	 (progn (setq pos (or (matlab-lattr-comm) (matlab-point-at-bol)))
		(goto-char pos)
		(if (and (> (current-column) (- fill-column matlab-fill-fudge))
			 (< (current-column) (+ fill-column matlab-fill-fudge)))
		    t
		  (goto-char orig)
		  nil))
	 ;; Now, lets find the nearest space (after or before fill column)
	 (let* ((after (save-excursion
			 (re-search-forward "[ \t]" nil t)))
		(before (save-excursion
			  (re-search-backward "[ \t]" nil t)))
		(afterd (- (or after (matlab-point-at-eol)) (point)))
		(befored (- (point) (or before (matlab-point-at-bol)))))
	   ;; Here, if "before" is actually the beginning of our
	   ;; indentation, then this is most obiously a bad place to
	   ;; break our lines.
	   (if before
	       (save-excursion
		 (goto-char before)
		 (if (<= (point) (save-excursion
				   (back-to-indentation)
				   (point)))
		     (setq before nil))))
	   (cond ((and after
		       (< afterd matlab-fill-fudge)
		       (< afterd befored))
		  (goto-char after)
		  t)
		 ((and before
		       (< befored matlab-fill-fudge)
		       (< befored afterd))
		  (goto-char before)
		  t)
		 (t (goto-char orig)
		    nil)))
	 ;; Now, lets find the nearest backwards
	 (progn
	   (re-search-backward "\\(\\s-\\|\\s.\\)+" nil t)
	   (while (and (looking-at "\\^\\|\\.\\|'")
		       (re-search-backward "\\(\\s-\\|\\s.\\)+" nil t)))
	   (if (or (not (looking-at "\\(\\s-\\|\\s.\\)+"))
		   (<= (point) (save-excursion
				 (back-to-indentation)
				 (point))))
	       (progn
		 ;; We failed in our mission to find anything, or fell
		 ;; of the edge of the earth.  If we are out of
		 ;; bounds, lets try again.
		 (goto-char orig)
		 (if (re-search-backward "\\s.+" nil t)
		     t
		   nil))
	     ;; Ok, we have a good location to break.  Check for column
	     ;; and ref against nearest list ending to predict a possibly
	     ;; better break point.
	     (forward-char 1)
	     (let ((okpos (current-column))
		   (startlst (save-excursion
			       (condition-case nil
				   (matlab-up-list -1)
				 (error nil))
			       (if (save-excursion
				     (forward-char -1)
				     (looking-at "\\w"))
				   (forward-word -1))
			       (current-column)))
		   (endlst (save-excursion
			     (condition-case nil
				 (matlab-up-list 1)
			       (error nil))
			     (current-column))))
	       ;; When evaluating list fudge factores, breaking on the
	       ;; edge of a list, or at the beginning of a function
	       ;; call can be more valuable than breaking on a symbol
	       ;; of a mid-sized list.  As such, allow double-fudge
	       ;; for lists.
	       (cond
		;; First, pick the end of a list.
		((and (< endlst matlab-fill-fudge-hard-maximum)
		      (<= endlst (+ fill-column matlab-fill-fudge))
		      (or (<= (* matlab-fill-fudge 2) (- endlst okpos))
			  (<= endlst fill-column))
		      (save-excursion
			(move-to-column endlst)
			(not (looking-at "\\^"))))
		 (move-to-column endlst)
		 t)
		;; Else, back up over this list and poke around
		((>= (* 2 matlab-fill-fudge) (- okpos startlst))
		 (move-to-column startlst)
		 t)
		;; Oh well, just do this symbol.
		(t (move-to-column okpos)
		   t)))))
	 ;; Well, this just sucks
	 (progn (goto-char orig)
		nil))))))

(defun matlab-auto-fill ()
  "Do auto filling.
Set variable `auto-fill-function' to this symbol to enable Matlab style auto
filling which will automatically insert `...' and the end of a line."
  (interactive)
  (let ((fill-prefix fill-prefix) ;; safe way of modifying fill-prefix.
	(fill-column (- fill-column
			(if matlab-fill-count-ellipsis-flag
			    (save-excursion
			      (move-to-column fill-column)
			      (if (not (bobp))
				  (forward-char -1))
			      (if (matlab-cursor-in-string 'incomplete)
				  4 3))
			  0))))
    (if (> (current-column) fill-column)
	(cond
	 ((matlab-ltype-comm-ignore)
	  nil)
	 ((or (matlab-ltype-comm)
	      (and (save-excursion (move-to-column fill-column)
				   (matlab-cursor-in-comment))
		   (matlab-lattr-comm)))
	  ;; If the whole line is a comment, do this.
	  (matlab-set-comm-fill-prefix) (do-auto-fill)
	  (matlab-reset-fill-prefix))
	 ((and (matlab-ltype-code)
	       (not (matlab-lattr-cont))
	       matlab-fill-code)
	  ;; If we are on a code line, we ellipsify before we fill.
	  (let ((m (make-marker)))
	    (move-marker m (point))
	    (set-marker-insertion-type m t)
	    (if (not (matlab-find-convenient-line-break))
		nil
	      (if (not (and (matlab-cursor-in-string 'incomplete)
			    (save-excursion
			      (forward-char -1)
			      (matlab-cursor-in-string 'incomplete))))
		  (progn
		    (delete-horizontal-space)
		    (insert " " matlab-elipsis-string "\n")
		    (matlab-indent-line))
		;; we are guaranteed to be in an incomplete string.
		(if matlab-fill-strings-flag
		    (let ((pos (point))
			  (pos2 nil))
		      (while (and (re-search-backward "'" nil t)
				  (progn (forward-char -1)
					 (looking-at "''"))))
		      (setq pos2 (point))
		      (if (not (looking-at "\\["))
			  (progn
			    (skip-chars-backward " \t")
			    (forward-char -1)))
		      (if (looking-at "\\[")
			  (goto-char pos)
			(goto-char pos2)
			(forward-char 1)
			(insert "[")
			(goto-char pos)
			(forward-char 1))
		      (delete-horizontal-space)
		      (insert "' " matlab-elipsis-string "\n")
		      (matlab-indent-line)
		      (insert "' ")))))
	    (goto-char m)))
	 ))))

(defun matlab-join-comment-lines ()
  "Join current comment line to the next comment line."
  ;; New w/ V2.0: This used to join the previous line, but I could find
  ;; no editors that had a "join" that did that.  I modified join to have
  ;; a behaviour I thought more inline with other editors.
  (interactive)
  (end-of-line)
  (if (looking-at "\n[ \t]*%")
      (replace-match " " t t nil)
    (error "No following comment to join with")))

(defun matlab-fill-region (beg-region end-region &optional justify-flag)
  "Fill the region between BEG-REGION and END-REGION.
Non-nil JUSTIFY-FLAG means justify comment lines as well."
  (interactive "*r\nP")
  (let ((end-reg-mk (make-marker)))
    (set-marker end-reg-mk end-region)
    (goto-char beg-region)
    (beginning-of-line)
    (while (< (point) end-reg-mk)
      ;; This function must also leave the point at the end of the
      ;; justified line.
      (matlab-fill-paragraph justify-flag)
      (forward-line 1)
      (beginning-of-line))))

(defun matlab-fill-comment-line (&optional justify)
  "Fill the current comment line.
With optional argument, JUSTIFY the comment as well."
  (interactive)
  (if (not (matlab-comment-on-line))
      (error "No comment to fill"))
  (beginning-of-line)
  ;; First, find the beginning of this comment...
  (while (and (looking-at matlab-cline-start-skip)
	      (not (bobp)))
    (forward-line -1)
    (beginning-of-line))
  (if (not (looking-at matlab-cline-start-skip))
      (forward-line 1))
  ;; Now scan to the end of this comment so we have our outer bounds,
  ;; and narrow to that region.
  (save-restriction
    (narrow-to-region (point)
		      (save-excursion
			(while (and (looking-at matlab-cline-start-skip)
				    (not (save-excursion (end-of-line) (eobp))))
			  (forward-line 1)
			  (beginning-of-line))
			(if (not (looking-at matlab-cline-start-skip))
			    (forward-line -1))
			(end-of-line)
			(point)))
    ;; Find the fill prefix...
    (matlab-comment-on-line)
    (looking-at "%[ \t]*")
    (let ((fill-prefix (concat (make-string (current-column) ? )
			       (match-string 0))))
      (fill-region (point-min) (point-max) justify))))

(defun matlab-justify-line ()
  "Delete space on end of line and justify."
  (interactive)
  (save-excursion
    (end-of-line)
    (delete-horizontal-space)
    (justify-current-line)))

(defun matlab-fill-paragraph (arg)
  "When in a comment, fill the current paragraph.
Paragraphs are always assumed to be in a comment.
ARG is passed to `fill-paragraph' and will justify the text."
  (interactive "P")
  (cond ((or (matlab-ltype-comm)
	     (and (matlab-cursor-in-comment)
		  (not (matlab-lattr-cont))))
	 ;; We are in a comment, lets fill the paragraph with some
	 ;; nice regular expressions.
	 (let ((paragraph-separate "%[a-zA-Z]\\|%[ \t]*$\\|[ \t]*$")
	       (paragraph-start "%[a-zA-Z]\\|%[ \t]*$\\|[ \t]*$")
	       (paragraph-ignore-fill-prefix nil)
	       (fill-prefix nil))
	   (matlab-set-comm-fill-prefix)
	   (fill-paragraph arg)))
	((matlab-ltype-code)
	 ;; Ok, lets get the outer bounds of this command, then
	 ;; completely refill it using the smart line breaking code.
	 (save-restriction
	   (narrow-to-region (save-excursion
			       (matlab-beginning-of-command)
			       (beginning-of-line)
			       (point))
			     (save-excursion
			       (matlab-end-of-command)
			       (point)))
	   ;; Remove all line breaks
	   (goto-char (point-min))
	   (while (and (re-search-forward "$" nil t)
		       (not (eobp)))
	     (delete-horizontal-space)
	     ;; Blow away continuation marks
	     (if (matlab-lattr-cont)
		 (progn
		   (goto-char (match-beginning 0))
		   (forward-char 1)
		   (delete-region (point) (matlab-point-at-eol))))
	     ;; Zap the CR
	     (if (not (eobp)) (delete-char 1))
	     ;; Clean up whitespace
	     (delete-horizontal-space)
	     ;; Clean up trailing comments
	     (if (and (looking-at "% *")
		      (matlab-cursor-in-comment))
		 (progn
		   (delete-char 1)
		   (delete-horizontal-space)))
	     (insert " "))
	   ;; Now fill till we are done
	   (goto-char (point-max))
	   (while (or (> (current-column) (+ fill-column matlab-fill-fudge))
		      (> (current-column) matlab-fill-fudge-hard-maximum))
	     (if (= (point)
		    (progn
		      (matlab-auto-fill)
		      (point)))
		 (error "Fill algorith failed!"))
	     (if arg (save-excursion
		       (forward-line -1)
		       (matlab-justify-line))))
	   (if arg (save-excursion
		     (forward-line -1)
		     (matlab-justify-line)))))
	(t
	 (message "Paragraph Fill not supported in this context."))))

;;; Semantic text insertion and management ====================================

(defun matlab-find-recent-variable-list (prefix)
  "Return a list of most recent variables starting with PREFIX as a string.
Reverse searches for the following are done first:
  1) Assignment
  2) if|for|while|switch <var>
  3) global variables
  4) function arguments.
All elements are saved in a list, which is then uniqafied.
If NEXT is non-nil, then the next element from the saved list is used.
If the list is empty, then searches continue backwards through the code."
  (matlab-navigation-syntax
    (let* ((bounds (save-excursion
		     (if (re-search-backward "^\\s-*function\\>" nil t)
			 (match-beginning 0) (point-min))))
	   (syms
	    (append
	     (save-excursion
	       (let ((lst nil))
		 (while (and
			 (re-search-backward
			  (concat "^\\s-*\\(" prefix "\\w+\\)\\s-*=")
			  bounds t)
			 (< (length lst) 10))
		   (setq lst (cons (match-string 1) lst)))
		 (nreverse lst)))
	     (save-excursion
	       (let ((lst nil))
		 (while (and (re-search-backward
			      (concat "\\<\\(" matlab-block-beg-pre-no-if
				      "\\)\\s-+(?\\s-*\\(" prefix
				      "\\w+\\)\\>")
			      bounds t)
			     (< (length lst) 10))
		   (setq lst (cons (match-string 2) lst)))
		 (nreverse lst)))
	     (save-excursion
	       (if (re-search-backward "^\\s-*global\\s-+" bounds t)
		   (let ((lst nil) m e)
		     (goto-char (match-end 0))
		     (while (looking-at "\\(\\w+\\)\\([ \t]+\\|$\\)")
		       (setq m (match-string 1)
			     e (match-end 0))
		       (if (equal 0 (string-match prefix m))
			   (setq lst (cons m lst)))
		       (goto-char e))
		     (nreverse lst))))
	     (save-excursion
	       (if (and (re-search-backward "^\\s-*function\\>" bounds t)
			(re-search-forward "\\<\\(\\w+\\)("
					   (matlab-point-at-eol) t))
		   (let ((lst nil) m e)
		     (while (looking-at "\\(\\w+\\)\\s-*[,)]\\s-*")
		       (setq m (match-string 1)
			     e (match-end 0))
		       (if (equal 0 (string-match prefix m))
			   (setq lst (cons m lst)))
		       (goto-char e))
		     (nreverse lst))))))
	   (fl nil))
      (while syms
	(if (car syms) (setq fl (cons (car syms) fl)))
	(setq syms (cdr syms)))
      (matlab-uniquafy-list (nreverse fl)))))

(defvar matlab-most-recent-variable-list nil
  "Maintained by `matlab-find-recent-variable'.")

(defun matlab-find-recent-variable (prefix &optional next)
  "Return the most recently used variable starting with PREFIX as a string.
See `matlab-find-recent-variable-list' for details.
In NEXT is non-nil, than continue through the list of elements."
  (if next
      (let ((next (car matlab-most-recent-variable-list)))
	(setq matlab-most-recent-variable-list
	      (cdr matlab-most-recent-variable-list))
	next)
    (let ((syms (matlab-find-recent-variable-list prefix))
	  (first nil))
      (if (eq matlab-completion-technique 'complete)
	  syms
	(setq first (car syms))
	(setq matlab-most-recent-variable-list (cdr syms))
	first))))

(defun matlab-find-user-functions-list (prefix)
  "Return a list of user defined functions that match PREFIX."
  (matlab-navigation-syntax
    (let ((syms
	   (append
	    (save-excursion
	      (goto-char (point-min))
	      (let ((lst nil))
		(while (re-search-forward "^\\s-*function\\>" nil t)
		  (if (re-search-forward
		       (concat "\\(" prefix "\\w+\\)\\s-*\\($\\|(\\)")
		       (matlab-point-at-eol) t)
		      (setq lst (cons (match-string 1) lst))))
		(nreverse lst)))
	    (let ((lst nil)
		  (files (directory-files
			  default-directory nil
			  (concat "^" prefix
				  "[a-zA-Z][a-zA-Z0-9_]+\\.m$"))))
	      (while files
		(setq lst (cons (progn (string-match "\\.m" (car files))
				       (substring (car files) 0
						  (match-beginning 0)))
				lst)
		      files (cdr files)))
	      lst)))
	  (fl nil))
      (while syms
	(if (car syms) (setq fl (cons (car syms) fl)))
	(setq syms (cdr syms)))
      (matlab-uniquafy-list (nreverse fl)))))

(defvar matlab-user-function-list nil
  "Maintained by `matlab-find-user-functions'.")

(defun matlab-find-user-functions (prefix &optional next)
  "Return a user function that match PREFIX and return it.
If optional argument NEXT is non-nil, then return the next found
object."
  (if next
      (let ((next (car matlab-user-function-list)))
	(setq matlab-user-function-list (cdr matlab-user-function-list))
	next)
    (let ((syms (matlab-find-user-functions-list prefix))
	  (first nil))
      (if (eq matlab-completion-technique 'complete)
	  syms
	(setq first (car syms))
	(setq matlab-user-function-list (cdr syms))
	first))))

(defvar matlab-generic-list-placeholder nil
  "Maintained by `matalb-generic-list-expand'.
Holds sub-lists of symbols left to be expanded.")

(defun matlab-generic-list-expand (list prefix &optional next)
  "Return an element from LIST that start with PREFIX.
If optional NEXT argument is non nil, then the next element in the
list is used.  nil is returned if there are not matches."
  (if next
      (let ((next (car matlab-generic-list-placeholder)))
	(setq matlab-generic-list-placeholder
	      (cdr matlab-generic-list-placeholder))
	next)
    (let ((re (concat "^" (regexp-quote prefix)))
	  (first nil)
	  (fl nil))
      (while list
	(if (string-match re (car list))
	    (setq fl (cons (car list) fl)))
	(setq list (cdr list)))
      (setq fl (nreverse fl))
      (if (eq matlab-completion-technique 'complete)
	  fl
	(setq first (car fl))
	(setq matlab-generic-list-placeholder (cdr fl))
	first))))

(defun matlab-solo-completions (prefix &optional next)
  "Return PREFIX matching elements for solo symbols.
If NEXT then the next patch from the list is used."
  (matlab-generic-list-expand matlab-keywords-solo prefix next))

(defun matlab-value-completions (prefix &optional next)
  "Return PREFIX matching elements for value symbols.
If NEXT then the next patch from the list is used."
  (matlab-generic-list-expand matlab-keywords-return prefix next))

(defun matlab-boolean-completions (prefix &optional next)
  "Return PREFIX matching elements for boolean symbols.
If NEXT then the next patch from the list is used."
  (matlab-generic-list-expand matlab-keywords-boolean prefix next))
 
(defun matlab-property-completions (prefix &optional next)
  "Return PREFIX matching elements for property names in strings.
If NEXT then the next property from the list is used."
  (let ((f (matlab-function-called-at-point))
	(lst matlab-property-lists)
	(foundlst nil)
	(expandto nil))
    ;; Look for this function.  If it is a known function then we
    ;; can now use a subset of available properties!
    (while (and lst (not foundlst))
      (if (string= (car (car lst)) f)
	  (setq foundlst (cdr (car lst))))
      (setq lst (cdr lst)))
    (if foundlst
	(setq foundlst (append foundlst matlab-core-properties))
      (setq foundlst matlab-all-known-properties))
    (setq expandto (matlab-generic-list-expand foundlst prefix next))
    ;; This looks to see if we have a singular completion.  If so,
    ;; then return it, and also append the "'" to the end.
    (cond ((and (listp expandto) (= (length expandto) 1))
	   (setq expandto (list (concat (car expandto) "'"))))
	  ((stringp expandto)
	   (setq expandto (concat expandto "'"))))
    expandto))

(defvar matlab-last-prefix nil
  "Maintained by `matlab-complete-symbol'.
The prefix used for the first completion command.")
(defvar matlab-last-semantic nil
  "Maintained by `matlab-complete-symbol'.
The last type of semantic used while completing things.")
(defvar matlab-completion-search-state nil
  "List of searching things we will be doing.")

(defun matlab-complete-symbol (&optional arg)
  "Complete a partially typed symbol in a Matlab mode buffer.
If the previously entered command was also `matlab-complete-symbol'
then undo the last completion, and find a new one.
  The types of symbols tried are based on the semantics of the current
cursor position.  There are two types of symbols.  For example, if the
cursor is in an if statement, boolean style functions and symbols are
tried first.  If the line is blank, then flow control, or high level
functions are tried first.
  The completion technique is controlled with `matlab-completion-technique'
It defaults to incremental completion described above.  If a
completion list is preferred, then change this to 'complete.  If you
just want a completion list once, then use the universal argument ARG
to change it temporarily."
  (interactive "P")
  (matlab-navigation-syntax
    (let* ((prefix (if (and (not (eq last-command 'matlab-complete-symbol))
			    (member (preceding-char) '(?  ?\t ?\n ?, ?\( ?\[ ?\')))
		       ""
		     (buffer-substring-no-properties
		      (save-excursion (forward-word -1) (point))
		      (point))))
	   (sem (matlab-lattr-semantics prefix))
	   (matlab-completion-technique
	    (if arg (cond ((eq matlab-completion-technique 'complete)
			   'increment)
			  (t 'complete))
	      matlab-completion-technique)))
      (if (not (eq last-command 'matlab-complete-symbol))
	  (setq matlab-last-prefix prefix
		matlab-last-semantic sem
		matlab-completion-search-state
		(cond ((eq sem 'solo)
		       '(matlab-solo-completions
			 matlab-find-user-functions
			 matlab-find-recent-variable))
		      ((eq sem 'boolean)
		       '(matlab-find-recent-variable
			 matlab-boolean-completions
			 matlab-find-user-functions
			 matlab-value-completions))
		      ((eq sem 'value)
		       '(matlab-find-recent-variable
			 matlab-find-user-functions
			 matlab-value-completions
			 matlab-boolean-completions))
		      ((eq sem 'property)
		       '(matlab-property-completions
			 matlab-find-user-functions
			 matlab-find-recent-variable
			 matlab-value-completions))
		      (t '(matlab-find-recent-variable
			   matlab-find-user-functions
			   matlab-value-completions
			   matlab-boolean-completions)))))
      (cond
       ((eq matlab-completion-technique 'increment)
	(let ((r nil) (donext (eq last-command 'matlab-complete-symbol)))
	  (while (and (not r) matlab-completion-search-state)
	    (message "Expand with %S" (car matlab-completion-search-state))
	    (setq r (funcall (car matlab-completion-search-state)
			     matlab-last-prefix donext))
	    (if (not r) (setq matlab-completion-search-state
			      (cdr matlab-completion-search-state)
			      donext nil)))
	  (delete-region (point) (progn (forward-char (- (length prefix)))
					(point)))
	  (if r
	      (insert r)
	    (insert matlab-last-prefix)
	    (message "No completions."))))
       ((eq matlab-completion-technique 'complete)
	(let ((allsyms (apply 'append
			      (mapcar (lambda (f) (funcall f prefix))
				      matlab-completion-search-state))))
	  (cond ((null allsyms)
		 (message "No completions.")
		 (ding))
		((= (length allsyms) 1)
		 (delete-region (point) (progn
					  (forward-char (- (length prefix)))
					  (point)))
		 (insert (car allsyms)))
		((= (length allsyms) 0)
		 (message "No completions."))
		(t
		 (let* ((al (mapcar (lambda (a) (list a)) allsyms))
			(c (try-completion prefix al)))
		   ;; This completion stuff lets us expand as much as is
		   ;; available to us. When the completion is the prefix
		   ;; then we want to display all the strings we've
		   ;; encountered.
		   (if (and (stringp c) (not (string= prefix c)))
		       (progn
			 (delete-region
			  (point)
			  (progn (forward-char (- (length prefix)))
				 (point)))
			 (insert c))
		     ;; `display-completion-list' does all the complex
		     ;; ui work for us.
		     (with-output-to-temp-buffer "*Completions*"
		       (display-completion-list
			(matlab-uniquafy-list allsyms)))))))))))))

(defun matlab-insert-end-block (&optional reindent)
  "Insert and END block based on the current syntax.
Optional argument REINDENT indicates if the specified block should be re-indented."
  (interactive "P")
  (if (not (matlab-ltype-empty)) (progn (end-of-line) (insert "\n")))
  (let ((valid t) (begin nil))
    (save-excursion
      (condition-case nil
	  (progn
	    (matlab-backward-sexp t)
	    (setq begin (point)
		  valid (buffer-substring-no-properties
			 (point) (save-excursion
				   (re-search-forward "[\n,;.]" nil t)
				   (point)))))
	(error (setq valid nil))))
    (if (not valid)
	(error "No block to end")
      (insert "end")
      (if (stringp valid) (insert " % " valid))
      (matlab-indent-line)
      (if reindent (indent-region begin (point) nil)))))

(tempo-define-template
 "matlab-for"
 '("for " p "=" p "," > n>
     r> &
     "end" > %)
 "for"
 "Insert a Matlab for statement"
 'matlab-tempo-tags
 )

(tempo-define-template
 "matlab-while"
 '("while (" p ")," > n>
     r> &
     "end" > %)
 "while"
 "Insert a Matlab while statement"
 'matlab-tempo-tags
 )

(tempo-define-template
 "matlab-if"
 '("if " p > n
     r>
     "end" > n)
 "if"
 "Insert a Matlab if statement"
 'matlab-tempo-tags
 )

(tempo-define-template
 "matlab-if-else"
 '("if " p > n
     r>
     "else" > n
     "end" > n)
 "if"
 "Insert a Matlab if statement"
 'matlab-tempo-tags
 )

(tempo-define-template
 "matlab-try"
 '("try " > n
     r>
     "catch" > n
     p > n
     "end" > n)
 "try"
 "Insert a Matlab try catch statement"
 'matlab-tempo-tags
 )

(tempo-define-template
 "matlab-switch"
 '("switch " p > n
     "otherwise" > n
     r>
     "end" > n)
 "switch"
 "Insert a Matlab switch statement with region in the otherwise clause."
 'matlab-tempo-tags)

(defun matlab-insert-next-case ()
  "Insert a case statement inside this switch statement."
  (interactive)
  ;; First, make sure we are where we think we are.
  (let ((valid t))
    (save-excursion
      (condition-case nil
	  (progn
	   (matlab-backward-sexp t)
	   (setq valid (looking-at "switch")))
	(error (setq valid nil))))
    (if (not valid)
	(error "Not in a switch statement")))
  (if (not (matlab-ltype-empty)) (progn (end-of-line) (insert "\n")))
  (indent-to 0)
  (insert "case ")
  (matlab-indent-line))

(tempo-define-template
 "matlab-function"
 '("function "
     (P "output argument(s): " output t)
     ;; Insert brackets only if there is more than one output argument
     (if (string-match "," (tempo-lookup-named 'output))
	 '(l "[" (s output) "]")
       '(l (s output)))
     ;; Insert equal sign only if there is output argument(s)
     (if (= 0 (length (tempo-lookup-named 'output))) nil
       " = ")
     ;; The name of a function, as defined in the first line, should
     ;; be the same as the name of the file without .m extension
     (if (= 1 (count-lines 1 (point)))
	 (tempo-save-named
	  'fname
	  (file-name-nondirectory (file-name-sans-extension
				   (buffer-file-name))))
       '(l (P "function name: " fname t)))
     (tempo-lookup-named 'fname)
     "("  (P "input argument(s): ") ")" n
     "% " (upcase (tempo-lookup-named 'fname)) " - " (P "H1 line: ") n
     "%   " p n)
 "function"
 "Insert a Matlab function statement"
 'matlab-tempo-tags
 )

(defun matlab-stringify-region (begin end)
  "Put Matlab 's around region, and quote all quotes in the string.
Stringification allows you to type in normal Matlab code, mark it, and
then turn it into a Matlab string that will output exactly what's in
the region.  BEGIN and END mark the region to be stringified."
  (interactive "r")
  (save-excursion
    (goto-char begin)
    (if (re-search-forward "\n" end t)
	(error
	 "You may only stringify regions that encompass less than one line"))
    (let ((m (make-marker)))
      (move-marker m end)
      (goto-char begin)
      (insert "'")
      (while (re-search-forward "'" m t)
	(insert "'"))
      (goto-char m)
      (insert "'"))))

(defun matlab-ispell-strings-region (begin end)
  "Spell check valid strings in region with Ispell.
Argument BEGIN and END mark the region boundary."
  (interactive "r")
  (require 'ispell)
  (save-excursion
    (goto-char begin)
    ;; Here we use the font lock function for finding strings.
    ;; Its cheap, fast, and accurate.
    (while (and (matlab-font-lock-string-match-normal end)
		(ispell-region (match-beginning 2) (match-end 2))))))

(defun matlab-ispell-strings ()
  "Spell check valid strings in the current buffer with Ispell.
Calls `matlab-ispell-strings-region'"
  (interactive)
  (matlab-ispell-strings-region (point-min) (point-max)))

(defun matlab-ispell-comments (&optional arg)
  "Spell check comments in the current buffer with Ispell.
Optional ARG means to only check the current comment."
  (interactive "P")
  (let ((beg (point-min))
	(end (point-max)))
  (if (and arg (matlab-ltype-comm))
      (setq beg (save-excursion (matlab-beginning-of-command) (point))
	    end (save-excursion (matlab-end-of-command) (point))))
  (save-excursion
    (goto-char beg)
    (beginning-of-line)
    (while (and (matlab-font-lock-comment-match end)
		(ispell-region (match-beginning 1) (match-end 1)))))))


;;; Block highlighting ========================================================

(defvar matlab-block-highlighter-timer nil
  "The timer representing the block highlighter.")

(defun matlab-enable-block-highlighting (&optional arg)
  "Start or stop the block highlighter.
Optional ARG is 1 to force enable, and -1 to disable.
If ARG is nil, then highlighting is toggled."
  (interactive "P")
  (if (not (fboundp 'matlab-run-with-idle-timer))
      (setq matlab-highlight-block-match-flag nil))
  ;; Only do it if it's enabled.
  (if (not matlab-highlight-block-match-flag)
      nil
    ;; Use post command idle hook as a local hook to dissuade too much
    ;; cpu time while doing other things.
    ;;(make-local-hook 'post-command-hook)
    (if (not arg)
	(setq arg
	      (if (member 'matlab-start-block-highlight-timer
			  post-command-hook)
		  -1 1)))
    (if (> arg 0)
	(add-hook 'post-command-hook 'matlab-start-block-highlight-timer)
      (remove-hook 'post-command-hook 'matlab-start-block-highlight-timer))))

(defvar matlab-block-highlight-overlay nil
  "The last highlighted overlay.")
(make-variable-buffer-local 'matlab-block-highlight-overlay)

(defvar matlab-block-highlight-timer nil
  "Last started timer.")
(make-variable-buffer-local 'matlab-block-highlight-timer)

(defun matlab-start-block-highlight-timer ()
  "Set up a one-shot timer if we are in Matlab mode."
  (if (eq major-mode 'matlab-mode)
      (progn
	(if matlab-block-highlight-overlay
	    (unwind-protect
		(matlab-delete-overlay matlab-block-highlight-overlay)
	      (setq matlab-block-highlight-overlay nil)))
	(if matlab-block-highlight-timer
	    (unwind-protect
		(matlab-cancel-timer matlab-block-highlight-timer)
	      (setq matlab-block-highlight-timer nil)))
	(setq matlab-block-highlight-timer
	      (matlab-run-with-idle-timer
	       1 nil 'matlab-highlight-block-match)))))
  
(defun matlab-highlight-block-match ()
  "Highlight a matching block if available."
  (setq matlab-block-highlight-timer nil)
  (let ((inhibit-quit nil)		;turn on G-g
	(matlab-scan-on-screen-only t))
    (if matlab-show-periodic-code-details-flag
	(matlab-show-line-info))
    (if (not (matlab-cursor-in-string-or-comment))
	(save-excursion
	  (if (or (bolp)
		  (looking-at "\\s-")
		  (save-excursion (forward-char -1) (looking-at "\\s-")))
	      nil
	    (forward-word -1))
	  (if (and (looking-at (concat (matlab-block-beg-re) "\\>"))
		   (not (looking-at "function")))
	      (progn
		;; We scan forward...
		(matlab-forward-sexp)
		(backward-word 1)
		(if (not (looking-at "end"))
		    nil ;(message "Unterminated block, or end off screen.")
		  (setq matlab-block-highlight-overlay
			(matlab-make-overlay (point)
					     (progn (forward-word 1)
						    (point))
					     (current-buffer)))
		  (matlab-overlay-put matlab-block-highlight-overlay
				      'face 'matlab-region-face)))
	    (if (and (looking-at (concat (matlab-block-end-pre) "\\>"))
		     (not (looking-at "function"))
		     (matlab-valid-end-construct-p))
		(progn
		  ;; We scan backward
		  (forward-word 1)
		  (condition-case nil
		      (progn
			(matlab-backward-sexp)
			(if (not (looking-at (matlab-block-beg-re)))
			    nil ;(message "Unstarted block at cursor.")
			  (setq matlab-block-highlight-overlay
				(matlab-make-overlay (point)
						     (progn (forward-word 1)
							    (point))
						     (current-buffer)))
			  (matlab-overlay-put matlab-block-highlight-overlay
					      'face 'matlab-region-face)))
		    (error (message "Unstarted block at cursor."))))
	      ;; do nothing
	      ))))))

;;; M Code verification & Auto-fix ============================================

(defun matlab-mode-verify-fix-file-fn ()
  "Verify the current buffer from `write-contents-hooks'."
  (if matlab-verify-on-save-flag
      (matlab-mode-verify-fix-file (> (point-max)
				      matlab-block-verify-max-buffer-size)))
  ;; Always return nil.
  nil)

(defun matlab-mode-verify-fix-file (&optional fast)
  "Verify the current buffer satisfies all M things that might be useful.
We will merely loop across a list of verifiers/fixers in
`matlab-mode-verify-fix-functions'.
If optional FAST is non-nil, do not perform usually lengthy checks."
  (interactive)
  (let ((p (point))
	(l matlab-mode-verify-fix-functions))
    (while l
      (funcall (car l) fast)
      (setq l (cdr l)))
    (goto-char p))
  (if (interactive-p)
      (message "Done.")))

;;
;; Add more auto verify/fix functions here!
;;
(defun matlab-mode-vf-functionname (&optional fast)
  "Verify/Fix the function name of this file.
Optional argument FAST is ignored."
  (matlab-navigation-syntax
    (goto-char (point-min))
    (while (and (or (matlab-ltype-empty) (matlab-ltype-comm))
		(/= (matlab-point-at-eol) (point-max)))
      (forward-line 1))
    (let ((func nil)
	  (bn (file-name-sans-extension
	       (file-name-nondirectory (buffer-file-name)))))
    (if (looking-at
	 ;; old function was too unstable.
	 ;;"\\(^function\\s-+\\)\\([^=\n]+=[ \t\n.]*\\)?\\(\\sw+\\)"
	 (concat "\\(^\\s-*function\\b[ \t\n.]*\\)\\(\\(\\[[^]]*\\]\\|\\sw+\\)"
		 "[ \t\n.]*=[ \t\n.]*\\)?\\(\\sw+\\)"))
	;; The expression above creates too many numeric matches
	;; to apply a known one to our function.  We cheat by knowing that
	;; match-end 0 is at the end of the function name.  We can then go
	;; backwards, and get the extents we need.  Navigation syntax
	;; lets us know that backward-word really covers the word.
	(let ((end (match-end 0))
	      (begin (progn (goto-char (match-end 0))
			    (forward-word -1)
			    (point))))
	  (setq func (buffer-substring begin end))
	  (if (not (string= func bn))
	      (if (not (matlab-mode-highlight-ask
			begin end
			"Function and file names are different. Fix?"))
		  nil
		(goto-char begin)
		(delete-region begin end)
		(insert bn))))))))

(defun matlab-mode-vf-block-matches-forward (&optional fast)
  "Verify/Fix unterminated (or un-ended) blocks.
This only checks block regions like if/end.
Optional argument FAST causes this check to be skipped."
  (goto-char (point-min))
  (let ((go t)
	(expr (concat "\\<\\(" (matlab-block-beg-pre) "\\)\\>")))
    (matlab-navigation-syntax
      (while (and (not fast) go (re-search-forward expr nil t))
	(forward-word -1)		;back over the special word
	(let ((s (point)))
	  (condition-case nil
	      (if (and (not (matlab-cursor-in-string-or-comment))
		       (not (looking-at "function")))
		  (progn
		    (matlab-forward-sexp)
		    (forward-word -1)
		    (if (not (looking-at "end\\>")) (setq go nil)))
		(forward-word 1))
	    (error (setq go nil)))
	  (if (and (not go) (goto-char s)
		   (not (matlab-mode-highlight-ask
			 (point) (save-excursion (forward-word 1) (point))
			 "Unterminated block.  Continue anyway?")))
	      (error "Unterminated Block found!")))
	(message "Block-check: %d%%" (/ (/ (* 100 (point)) (point-max)) 2))))))
  
(defun matlab-mode-vf-block-matches-backward (&optional fast)
  "Verify/fix unstarted (or dangling end) blocks.
Optional argument FAST causes this check to be skipped."
  (goto-char (point-max))
  (let ((go t) (expr (concat "\\<\\(" (matlab-block-end-no-function-re)
			     "\\)\\>")))
    (matlab-navigation-syntax
      (while (and (not fast) go (re-search-backward expr nil t))
	(forward-word 1)
	(let ((s (point)))
	  (condition-case nil
	      (if (and (not (matlab-cursor-in-string-or-comment))
		       (matlab-valid-end-construct-p))
		  (matlab-backward-sexp)
		(backward-word 1))
	    (error (setq go nil)))
	  (if (and (not go) (goto-char s)
		   (not (matlab-mode-highlight-ask
			 (point) (save-excursion (backward-word 1) (point))
			 "Unstarted block.  Continue anyway?")))
	      (error "Unstarted Block found!")))
	(message "Block-check: %d%%"
		 (+ (/ (/ (* 100 (- (point-max) (point))) (point-max)) 2) 50))))))

;;; Utility for verify/fix actions if you need to highlight
;;  a section of the buffer for the user's approval.
(defun matlab-mode-highlight-ask (begin end prompt)
  "Highlight from BEGIN to END while asking PROMPT as a yes-no question."
  (let ((mo (matlab-make-overlay begin end (current-buffer)))
	(ans nil))
    (condition-case nil
	(progn
	  (matlab-overlay-put mo 'face 'matlab-region-face)
	  (setq ans (y-or-n-p prompt))
	  (matlab-delete-overlay mo))
      (quit (matlab-delete-overlay mo) (error "Quit")))
    ans))

;;; Quietafy an M file to remove accidental display of ANS during a run.
;;  Useful if you have random outputs and you don't know where they are from,
;;  or before compiling to standalone where some functions now have outputs
;;  that did not have outputs earlier.
;;
;;  You probably don't want this as a default verify function
(defun matlab-mode-vf-quietafy-buffer (&optional fast)
  "Find all commands that do not end in ;, and add one.
This has the effect of removing any extraneous output that may not be
desired.  Optional argument FAST is not used."
  (interactive)
  (save-excursion
    (beginning-of-buffer)
    (let ((msgpos 0) (dir .2))
      (while (not (save-excursion (end-of-line) (eobp)))
	(message (aref [ "Scanning o...." "Scanning .o..." "Scanning ..o.."
			 "Scanning ...o." "Scanning ....o" ] (floor msgpos)))
	(setq msgpos (+ msgpos dir))
	(if (or (> msgpos 5) (< msgpos 0)) (setq dir (- dir)
						 msgpos (+ (* 2 dir) msgpos)))
	(matlab-end-of-command (point))
	(if (matlab-cursor-in-comment)
	    (progn
	      (matlab-comment-on-line)
	      (skip-chars-backward " \t")))
	(if (and (not (= (preceding-char) ?\;))
		 (not (matlab-cursor-in-string t))
		 (not (save-excursion
			(beginning-of-line)
			(looking-at
			 "\\s-*\\(function\\|for\\|while\\|try\\|catch\\|\
switch\\|otherwise\\|case\\|break\\|if\\|else\\|end\\|return\\|disp\\|\
$\\|%\\)"))))
	    (let ((p (point)))
	      (skip-chars-backward " \t")
	      (if (/= p (point))
		  (progn
		    (delete-region p (point))
		    (forward-line -1))
		(if (matlab-mode-highlight-ask (point) (+ 1 (point))
					       "Add Semi colon here? ")
		    (insert ";")))))
	(forward-line 1))))
  (message "Scanning .... done"))
  


;;; V19 stuff =================================================================

(defun matlab-mode-hilit ()
  "Set up hilit19 support for `matlab-mode'."
  (interactive)
  (cond (window-system
	 (setq hilit-mode-enable-list  '(not text-mode)
	       hilit-background-mode   'light
	       hilit-inhibit-hooks     nil
	       hilit-inhibit-rebinding nil)
	 (require 'hilit19)
	 (hilit-set-mode-patterns 'matlab-mode matlab-hilit19-patterns))))

(defvar matlab-mode-menu-keymap nil
  "Keymap used in Matlab mode to provide a menu.")

(defun matlab-frame-init ()
  "Initialize Emacs 19+ menu system."
  (interactive)
  ;; make a menu keymap
  (easy-menu-define
   matlab-mode-menu
   matlab-mode-map
   "Matlab menu"
   '("Matlab"
     ["Start Matlab" matlab-shell t]
     ["Save and go" matlab-shell-save-and-go t]
     ["Run Region" matlab-shell-run-region t]
     ["Version" matlab-show-version t]
     "----"
     ["Find M file" matlab-find-file-on-path t]
     ("Auto Fix"
      ["Verify/Fix source" matlab-mode-verify-fix-file t]
      ["Spell check strings" matlab-ispell-strings t]
      ["Spell check comments" matlab-ispell-comments t]
      ["Quietafy source" matlab-mode-vf-quietafy-buffer t])
     ("Navigate"
      ["Beginning of Command" matlab-beginning-of-command t]
      ["End of Command" matlab-end-of-command t]
      ["Forward Block" matlab-forward-sexp t]
      ["Backward Block" matlab-backward-sexp t]
      ["Beginning of Function" matlab-beginning-of-defun t]
      ["End of Function" matlab-end-of-defun t])
     ("Format"
      ["Justify Line" matlab-justify-line t]
      ["Fill Region" matlab-fill-region t]
      ["Fill Comment Paragraph" matlab-fill-paragraph
       (save-excursion (matlab-comment-on-line))]
      ["Join Comment" matlab-join-comment-lines
       (save-excursion (matlab-comment-on-line))]
      ["Comment Region" matlab-comment-region t])
     ("Insert"
      ["Complete Symbol" matlab-complete-symbol t]
      ["Comment" matlab-comment t]
      ["if end" tempo-template-matlab-if t]
      ["if else end" tempo-template-matlab-if-else t]
      ["for end" tempo-template-matlab-for t]
      ["switch otherwise end" tempo-template-matlab-switch t]
      ["Next case" matlab-insert-next-case t]
      ["try catch end" tempo-template-matlab-try t]
      ["while end" tempo-template-matlab-while t]
      ["End of block" matlab-insert-end-block t]
      ["Function" tempo-template-matlab-function t]
      ["Stringify Region" matlab-stringify-region t]
      )
     ("Customize"
;      ["Auto Fill Counts Elipsis"
;       (lambda () (setq matlab-fill-count-ellipsis-flag
;			(not matlab-fill-count-ellipsis-flag)))
;       :style toggle :selected 'matlab-fill-count-ellipsis-flag]
      ["Indent Function Body"
       (setq matlab-indent-function (not matlab-indent-function))
       :style toggle :selected matlab-indent-function]
      ["Verify File on Save"
       (setq matlab-verify-on-save-flag (not matlab-verify-on-save-flag))
       :style toggle :selected matlab-verify-on-save-flag]
      ["Auto Fill does Code"
       (setq matlab-fill-code (not matlab-fill-code))
       :style toggle :selected matlab-fill-code ]
      ["Periodic Code Details"
       (setq matlab-show-periodic-code-details-flag
	     (not matlab-show-periodic-code-details-flag))
       :style toggle :selected matlab-show-periodic-code-details-flag ]
      ["Highlight Matching Blocks"
       (matlab-enable-block-highlighting)
       :style toggle :selected (member 'matlab-start-block-highlight-timer
				       post-command-hook) ]
      
      ["Customize" (lambda () (interactive) (customize-group 'matlab))
       (and (featurep 'custom) (fboundp 'custom-declare-variable)) ]
      )
     "----"
     ["Run M Command" matlab-shell-run-command (matlab-shell-active-p)]
     ["Describe Command" matlab-shell-describe-command (matlab-shell-active-p)]
     ["Describe Variable" matlab-shell-describe-variable (matlab-shell-active-p)]
     ["Command Apropos" matlab-shell-apropos (matlab-shell-active-p)]
     ["Topic Browser" matlab-shell-topic-browser (matlab-shell-active-p)]
     ))
  (easy-menu-add matlab-mode-menu matlab-mode-map))

;;; Matlab shell =============================================================

(defgroup matlab-shell nil
  "Matlab shell mode."
  :prefix "matlab-shell-"
  :group 'matlab)

(defcustom matlab-shell-command "matlab"
  "*The name of the command to be run which will start the Matlab process."
  :group 'matlab-shell
  :type 'string)

(defcustom matlab-shell-command-switches ""
  "*Command line parameters run with `matlab-shell-command'."
  :group 'matlab-shell
  :type '(string :tag "Switch: "))

(defcustom matlab-shell-echoes t
  "*If `matlab-shell-command' echoes input."
  :group 'matlab-shell
  :type 'boolean)

(defcustom matlab-shell-enable-gud-flag t
  "*Non-nil means to use GUD mode when running the Matlab shell."
  :group 'matlab-shell
  :type 'boolean)

(defcustom matlab-shell-mode-hook nil
  "*List of functions to call on entry to Matlab shell mode."
  :group 'matlab-shell
  :type 'hook)

(defvar matlab-shell-buffer-name "Matlab"
  "Name used to create `matlab-shell' mode buffers.
This name will have *'s surrounding it.")

(defun matlab-shell-active-p ()
  "Return t if the Matlab shell is active."
  (if (get-buffer (concat "*" matlab-shell-buffer-name "*"))
      (save-excursion
	(set-buffer (concat "*" matlab-shell-buffer-name "*"))
	(if (comint-check-proc (current-buffer))
	    (current-buffer)))))

(defvar matlab-shell-mode-map ()
  "Keymap used in `matlab-shell-mode'.")

(defvar matlab-shell-font-lock-keywords-1
  (append matlab-font-lock-keywords matlab-shell-font-lock-keywords)
  "Keyword symbol used for font-lock mode.")

(defvar matlab-shell-font-lock-keywords-2
  (append matlab-shell-font-lock-keywords-1 matlab-gaudy-font-lock-keywords)
  "Keyword symbol used for gaudy font-lock symbols.")

(defvar matlab-shell-font-lock-keywords-3
  (append matlab-shell-font-lock-keywords-2
	  matlab-really-gaudy-font-lock-keywords)
  "Keyword symbol used for really gaudy font-lock symbols.")

(eval-when-compile (require 'gud) (require 'comint) (require 'shell))

;;;###autoload
(defun matlab-shell ()
  "Create a buffer with Matlab running as a subprocess.

Matlab shell cannot work on the MS Windows platform because Matlab is not
a console application."
  (interactive)
  ;; Matlab shell does not work by default on the Windows platform.  Only
  ;; permit it's operation when the shell command string is different from
  ;; the default value.  (True when the engine program is running.)
  (if (and (or (eq window-system 'pc) (eq window-system 'w32))
	   (string= matlab-shell-command "matlab"))
      (error "Matlab cannot be run as a inferior process.  \
Try C-h f matlab-shell RET"))

  (require 'shell)
  (require 'gud)

  ;; Make sure this is safe...
  (if (and matlab-shell-enable-gud-flag (fboundp 'gud-def))
      ;; We can continue using GUD
      nil
    (message "Sorry, your emacs cannot use the Matlab Shell GUD features.")
    (setq matlab-shell-enable-gud-flag nil))

  (switch-to-buffer (concat "*" matlab-shell-buffer-name "*"))
  (if (matlab-shell-active-p)
      nil
    ;; Build keymap here in case someone never uses comint mode
    (if matlab-shell-mode-map
	()
      (setq matlab-shell-mode-map
	    (let ((km (make-sparse-keymap 'matlab-shell-mode-map)))
	      (if (fboundp 'set-keymap-parent)
		  (set-keymap-parent km comint-mode-map)
		;; 19.31 doesn't have set-keymap-parent
		(setq km (nconc km comint-mode-map)))
	      (substitute-key-definition 'next-error 'matlab-shell-last-error
					 km global-map)
	      (define-key km [(control h) (control m)]
		matlab-help-map)
	      (define-key km [(tab)]
		'comint-dynamic-complete-filename)
	      (define-key km [(control up)]
		'comint-previous-matching-input-from-input)
	      (define-key km [(control down)]
		'comint-next-matching-input-from-input)
	      (define-key km [up]
		'comint-previous-matching-input-from-input)
	      (define-key km [down]
		'comint-next-matching-input-from-input)
	      (define-key km [(control return)] 'comint-kill-input)
	      (define-key km "\C-?"
		'matlab-shell-delete-backwards-no-prompt)
	      (define-key km [(backspace)]
		'matlab-shell-delete-backwards-no-prompt)
	      km)))
    (switch-to-buffer
     (make-comint matlab-shell-buffer-name matlab-shell-command
		  nil matlab-shell-command-switches))
    
    (setq shell-dirtrackp t)
    (comint-mode)

    (if matlab-shell-enable-gud-flag
	(progn
	  (gud-mode)
	  (make-local-variable 'gud-marker-filter)
	  (setq gud-marker-filter 'gud-matlab-marker-filter)
	  (make-local-variable 'gud-find-file)
	  (setq gud-find-file 'gud-matlab-find-file)

	  (set-process-filter (get-buffer-process (current-buffer))
			      'gud-filter)
	  (set-process-sentinel (get-buffer-process (current-buffer))
				'gud-sentinel)
	  (gud-set-buffer))
      ;; What to do when there is no GUD
      ;(set-process-filter (get-buffer-process (current-buffer))
	;		  'matlab-shell-process-filter)
      )

    ;; Comint and GUD both try to set the mode.  Now reset it to
    ;; matlab mode.
    (matlab-shell-mode)))

(defcustom matlab-shell-logo
  (if (fboundp 'locate-data-file)
      ;; Starting from XEmacs 20.4 use locate-data-file
      (locate-data-file "matlab.xpm")
    (expand-file-name "matlab.xpm" data-directory))
  "*The Matlab logo file."
  :group 'matlab-shell
  :type '(choice (const :tag "None" nil)
		 (file :tag "File" "")))

 
(defun matlab-shell-hack-logo (str)
  "Replace the text logo with a real logo.
STR is passed from the commint filter."
  (when (string-match "< M A T L A B (R) >" str)
    (save-excursion
      (when (re-search-backward "^[ \t]+< M A T L A B (R) >" (point-min) t)
 	(delete-region (match-beginning 0) (match-end 0))
 	(insert (make-string 16 ? ))
 	(set-extent-begin-glyph (make-extent (point) (point))
 				(make-glyph matlab-shell-logo))))
    ;; Remove this function from `comint-output-filter-functions'
    (remove-hook 'comint-output-filter-functions
 		 'matlab-shell-hack-logo)))

(defun matlab-shell-mode ()
  "Run Matlab as a subprocess in an Emacs buffer.

This mode will allow standard Emacs shell commands/completion to occur
with Matlab running as an inferior process.  Additionally, this shell
mode is integrated with `matlab-mode', a major mode for editing M
code.

> From an M file buffer:
\\<matlab-mode-map>
\\[matlab-shell-save-and-go] - Save the current M file, and run it in a \
Matlab shell.

> From Shell mode:
\\<matlab-shell-mode-map>
\\[matlab-shell-last-error] - find location of last Matlab runtime error \
in the offending M file.

> From an M file, or from Shell mode:
\\<matlab-mode-map>
\\[matlab-shell-run-command] - Run COMMAND and show result in a popup buffer.
\\[matlab-shell-describe-variable] - Show variable contents in a popup buffer.
\\[matlab-shell-describe-command] - Show online documentation for a command \
in a popup buffer.
\\[matlab-shell-apropos] - Show output from LOOKFOR command in a popup buffer.
\\[matlab-shell-topic-browser] - Topic browser using HELP.

> Keymap:
\\{matlab-mode-map}"
  (setq major-mode 'matlab-shell-mode
	mode-name "M-Shell"
	comint-prompt-regexp "^K?>> *"
	comint-delimiter-argument-list (list [ 59 ]) ; semi colon
	comint-dynamic-complete-functions '(comint-replace-by-expanded-history)
	comint-process-echoes matlab-shell-echoes
	)
  ;;(add-hook 'comint-input-filter-functions 'shell-directory-tracker)
  ;; Add a spiffy logo if we are running XEmacs
  (if (and (string-match "XEmacs" emacs-version)
	   (stringp matlab-shell-logo)
	   (file-readable-p matlab-shell-logo))
      (add-hook 'comint-output-filter-functions 'matlab-shell-hack-logo))
  (make-local-variable 'comment-start)
  (setq comment-start "%")
  (use-local-map matlab-shell-mode-map)
  (set-syntax-table matlab-mode-syntax-table)
  (make-local-variable 'font-lock-defaults)
  (setq font-lock-defaults '((matlab-shell-font-lock-keywords-1
			      matlab-shell-font-lock-keywords-2
			      matlab-shell-font-lock-keywords-3)
			     t nil ((?_ . "w"))))
  (make-local-variable 'gud-marker-acc)
  (easy-menu-define
   matlab-shell-menu
   matlab-shell-mode-map
   "Matlab shell menu"
   '("Matlab"
     ["Goto last error" matlab-shell-last-error t]
     "----"
     ["Stop On Errors" matlab-shell-dbstop-error t]
     ["Don't Stop On Errors" matlab-shell-dbclear-error t]
     "----"
     ["Run Command" matlab-shell-run-command t]
     ["Describe Variable" matlab-shell-describe-variable t]
     ["Describe Command" matlab-shell-describe-command t]
     ["Lookfor Command" matlab-shell-apropos t]
     ["Topic Browser" matlab-shell-topic-browser t]
     "----"
     ["Demos" matlab-shell-demos t]
     ["Close Current Figure" matlab-shell-close-current-figure t]
     ["Close Figures" matlab-shell-close-figures t]
     "----"
     ["Exit" matlab-shell-exit t]))
  (easy-menu-add matlab-shell-menu matlab-shell-mode-map)
  
  (if matlab-shell-enable-gud-flag
      (progn
	(gud-def gud-break  "dbstop at %l in %f"  "\C-b" "Set breakpoint at current line.")
	(gud-def gud-remove "dbclear at %l in %f" "\C-d" "Remove breakpoint at current line")
	(gud-def gud-step   "dbstep %p"           "\C-s" "Step one source line with display.")
	(gud-def gud-cont   "dbcont"              "\C-r" "Continue with display.")
	(gud-def gud-finish "dbquit"              "\C-f" "Finish executing current function.")
	(gud-def gud-up     "dbup %p"             "<"    "Up N stack frames (numeric arg).")
	(gud-def gud-down   "dbdown %p"           ">"    "Down N stack frames (numeric arg).")
	(gud-def gud-print  "%e"                  "\C-p" "Evaluate M expression at point.")
	(if (fboundp 'gud-make-debug-menu)
	    (gud-make-debug-menu))
	(if (fboundp 'gud-overload-functions)
	    (gud-overload-functions
	     '((gud-massage-args . gud-matlab-massage-args)
	       (gud-marker-filter . gud-matlab-marker-filter)
	       (gud-find-file . gud-matlab-find-file))))
	;; XEmacs doesn't seem to have this concept already.  Oh well.
	(setq gud-marker-acc nil)
	;; XEmacs has problems w/ this variable.  Set it here.
	(set-marker comint-last-output-start (point-max))
	))
  (run-hooks 'matlab-shell-mode-hook)
  (matlab-show-version)
  )

(defvar gud-matlab-marker-regexp-1 "^K>>"
  "Regular expression for finding a file line-number.")

(defvar gud-matlab-marker-regexp-2
  "^> In \\([-.a-zA-Z0-9_/@]+\\) \\((\\w+) \\|\\)at line \\([0-9]+\\)[ \n]+"
  "Regular expression for finding a file line-number.
Please note: The > character represents the current stack frame, so if there
are several frames, this makes sure we pick the right one to popup.")

(defvar gud-matlab-marker-regexp-prefix "^> In "
  "A prefix to scan for to know if output might be scarfed later.")

;; The regular expression covers the following form:
;; Errors:  Error in ==> <filename>
;;          On line # ==> <command_name>
;; Syntax:  Syntax error in ==> <filename>
;;          On line # ==> <sample-text>
;; Warning: In <filename> at line # <stuff>
(defvar gud-matlab-error-regexp
  (concat "\\(Error in ==>\\|Syntax error in ==>\\|In\\) "
	  "\\([-.a-zA-Z_0-9/ \\\\:]+\\).*[\n ][Oa][nt] line "
	  "\\([0-9]+\\) ?")
  "Regular expression finding where an error occurred.")

(defvar matlab-last-frame-returned nil
  "Store the previously returned frame for Matlabs difficult debugging output.
It is reset to nil whenever we are not prompted by the K>> output.")

(defvar matlab-one-db-request nil
  "Set to t if we requested a debugger command trace.")

(defun gud-matlab-massage-args (file args)
  "Argument massager for starting matlab file.
I don't think I have to do anything, but I'm not sure.
FILE is ignored, and ARGS is returned."
  args)

(defun gud-matlab-marker-filter (string)
  "Filters STRING for the Unified Debugger based on Matlab output.
Swiped ruthlessly from GDB mode in gud.el"
  (let ((garbage (concat "\\(" (regexp-quote "\C-g") "\\|"
 			 (regexp-quote "\033[H0") "\\|"
 			 (regexp-quote "\033[H\033[2J") "\\|"
 			 (regexp-quote "\033H\033[2J") "\\)")))
    (while (string-match garbage string)
      (if (= (aref string (match-beginning 0)) ?\C-g)
	  (beep t))
      (setq string (replace-match "" t t string))))
  (setq gud-marker-acc (concat gud-marker-acc string))
  (let ((output "") (frame nil))

    ;; Remove output from one stack trace...
    (if (eq matlab-one-db-request t)
	(if (string-match "db[a-z]+[ \n]+" gud-marker-acc)
	    (setq gud-marker-acc (substring gud-marker-acc (match-end 0))
		  matlab-one-db-request 'prompt)))

    ;; Process all the complete markers in this chunk.
    (while (and (not (eq matlab-one-db-request t))
		(string-match gud-matlab-marker-regexp-2 gud-marker-acc))

      (setq

       ;; Extract the frame position from the marker.
       frame (cons (match-string 1 gud-marker-acc)
		   (string-to-int (substring gud-marker-acc
					     (match-beginning 3)
					     (match-end 3))))

       ;; Append any text before the marker to the output we're going
       ;; to return - we don't include the marker in this text.
       ;; If this is not a requested piece of text, then include
       ;; it into the output.
       output (concat output
		      (substring gud-marker-acc 0
				 (if matlab-one-db-request
				     (match-beginning 0)
				   (match-end 0))))

       ;; Set the accumulator to the remaining text.
       gud-marker-acc (substring gud-marker-acc (match-end 0))))

    (if frame
	(progn
	  ;; We have a frame, so we don't need to do extra checking.
	  (setq matlab-last-frame-returned frame)
	  )
      (if (and (not matlab-one-db-request)
	       (string-match gud-matlab-marker-regexp-1 gud-marker-acc))
	  (progn
	    ;; Here we know we are in debug mode, so find our stack, and
	    ;; deal with that later...
	    (setq matlab-one-db-request t)
	    (process-send-string (get-buffer-process gud-comint-buffer)
				 "dbstack\n"))))

    ;; Check for a prompt to nuke...
    (if (and (eq matlab-one-db-request 'prompt)
	     (string-match "^K?>> $" gud-marker-acc))
	(setq matlab-one-db-request nil
	      output ""
	      gud-marker-acc (substring gud-marker-acc (match-end 0))))

    ;; Finish off this part of the output.  None of our special stuff
    ;; ends with a \n, so display those as they show up...
    (while (string-match "^[^\n]*\n" gud-marker-acc)
      (setq output (concat output (substring gud-marker-acc 0 (match-end 0)))
	    gud-marker-acc (substring gud-marker-acc (match-end 0))))

    (if (string-match gud-matlab-marker-regexp-prefix gud-marker-acc)
	;; We could be collecting something.  Wait for a while.
	nil
      (setq output (concat output gud-marker-acc)
	    gud-marker-acc "")
      ;; Check our output for a prompt, and existence of a frame.
      ;; If t his is true, throw out the debug arrow stuff.
      (if (and (string-match "^>> $" output)
	       gud-last-last-frame)
	  (progn
	    (setq overlay-arrow-position nil
		  gud-last-last-frame nil)
	    (sit-for 0)
	    )))

    (if frame (setq gud-last-frame frame))

    ;;(message "[%s] [%s]" output gud-marker-acc)

    output))

(defun gud-matlab-find-file (f)
  "Find file F when debugging frames in Matlab."
  (save-excursion
    (let* ((realfname (if (string-match "\\.\\(p\\)$" f)
			  (progn
			    (aset f (match-beginning 1) ?m)
			    f)
			f))
	   (buf (find-file-noselect realfname)))
      (set-buffer buf)
      (if (fboundp 'gud-make-debug-menu)
	  (gud-make-debug-menu))
      buf)))

(defun matlab-shell-delete-backwards-no-prompt (&optional arg)
  "Delete one char backwards without destroying the matlab prompt.
Optional argument ARG describes the number of chars to delete."
  (interactive "P")
  (let ((promptend (save-excursion
		     (beginning-of-line)
		     (if (looking-at "K?>> ")
			 (match-end 0)
		       (point))))
	(numchars (if (integerp arg) (- arg) -1)))
    (if (<= promptend (+ (point) numchars))
	(delete-char numchars)
      (error "Beginning of line"))))


;;; Matlab mode Shell commands ================================================

(defun matlab-show-matlab-shell-buffer ()
  "Switch to the buffer containing the matlab process."
  (interactive)
  (let ((msbn (concat "*" matlab-shell-buffer-name "*")))
    (if (get-buffer msbn)
	(switch-to-buffer-other-window msbn)
      (message "There is not an active Matlab process."))))

(defvar matlab-shell-save-and-go-history '("()")
  "Keep track of parameters passed to the Matlab shell.")

(defun matlab-shell-save-and-go ()
  "Save this M file, and evaluate it in a Matlab shell."
  (interactive)
  (if (not (eq major-mode 'matlab-mode))
      (error "Save and go is only useful in a Matlab buffer!"))
  (let ((fn-name (file-name-sans-extension
		  (file-name-nondirectory (buffer-file-name))))
	(msbn (concat "*" matlab-shell-buffer-name "*"))
	(param ""))
    (save-buffer)
    ;; Do we need parameters?
    (if (save-excursion
	  (goto-char (point-min))
	  (end-of-line)
	  (forward-sexp -1)
	  (looking-at "([a-zA-Z]"))
	(setq param (read-string "Parameters: "
				 (car matlab-shell-save-and-go-history)
				 'matlab-shell-save-and-go-history)))
    ;; No buffer?  Make it!
    (if (not (get-buffer msbn)) (matlab-shell))
    ;; Ok, now fun the function in the matlab shell
    (if (get-buffer-window msbn t)
	(select-window (get-buffer-window msbn t))
      (switch-to-buffer (concat "*" matlab-shell-buffer-name "*")))
    (matlab-shell-send-string (concat fn-name " " param "\n"))))

(defun matlab-shell-run-region (beg end)
  "Run region from BEG to END and display result in Matlab shell.
This command requires an active Matlab shell."
  (interactive "r")
  (if (> beg end) (let (mid) (setq mid beg beg end end mid)))
  (let ((command (let ((str (concat (buffer-substring-no-properties beg end)
 				    "\n")))
 		   (while (string-match "\n\\s-*\n" str)
 		     (setq str (concat (substring str 0 (match-beginning 0))
 				       "\n"
 				       (substring str (match-end 0)))))
 		   str))
 	(msbn (matlab-shell-buffer-barf-not-running))
 	(lastcmd))
    (save-excursion
      (set-buffer msbn)
      (if (not (matlab-on-prompt-p))
 	  (error "Matlab shell must be non-busy to do that"))
      ;; Save the old command
      (beginning-of-line)
      (re-search-forward comint-prompt-regexp)
      (setq lastcmd (buffer-substring (point) (matlab-point-at-eol)))
      (delete-region (point) (matlab-point-at-eol))
      ;; We are done error checking, run the command.
      (matlab-shell-send-string command)
      (insert lastcmd))
    (set-buffer msbn)
    (goto-char (point-max))
    (display-buffer msbn)
    ))
 

;;; Matlab Shell Commands =====================================================

(defun matlab-read-word-at-point ()
  "Get the word closest to point, but do not change position.
Has a preference for looking backward when not directly on a symbol.
Snatched and hacked from dired-x.el"
  (let ((word-chars "a-zA-Z0-9_")
	(bol (matlab-point-at-bol))
	(eol (matlab-point-at-eol))
        start)
    (save-excursion
      ;; First see if just past a word.
      (if (looking-at (concat "[" word-chars "]"))
	  nil
	(skip-chars-backward (concat "^" word-chars "{}()\[\]") bol)
	(if (not (bobp)) (backward-char 1)))
      (if (numberp (string-match (concat "[" word-chars "]")
				 (char-to-string (following-char))))
          (progn
            (skip-chars-backward word-chars bol)
            (setq start (point))
            (skip-chars-forward word-chars eol))
        (setq start (point)))		; If no found, return empty string
      (buffer-substring start (point)))))

(defun matlab-read-line-at-point ()
  "Get the line under point, if command line."
  (if (eq major-mode 'matlab-shell-mode)
      (save-excursion
	(beginning-of-line)
	(if (not (looking-at (concat comint-prompt-regexp)))
	    ""
	  (search-forward-regexp comint-prompt-regexp)
	  (buffer-substring (point) (matlab-point-at-eol))))
    (save-excursion
      ;; In matlab buffer, find all the text for a command.
      ;; so back over until there is no more continuation.
      (while (save-excursion (forward-line -1) (matlab-lattr-cont))
	(forward-line -1))
      ;; Go forward till there is no continuation
      (beginning-of-line)
      (let ((start (point)))
	(while (matlab-lattr-cont) (forward-line 1))
	(end-of-line)
	(buffer-substring start (point))))))

(defun matlab-non-empty-lines-in-string (str)
  "Return number of non-empty lines in STR."
  (let ((count 0)
	(start 0))
    (while (string-match "^.+$" str start)
      (setq count (1+ count)
	    start (match-end 0)))
    count))

(defun matlab-output-to-temp-buffer (buffer output)
  "Print output to temp buffer, or a message if empty string.
BUFFER is the buffer to output to, and OUTPUT is the text to insert."
  (let ((lines-found (matlab-non-empty-lines-in-string output)))
    (cond ((= lines-found 0)
	   (message "(Matlab command completed with no output)"))
	  ((= lines-found 1)
	   (string-match "^.+$" output)
	   (message (substring output (match-beginning 0)(match-end 0))))
	  (t (with-output-to-temp-buffer buffer (princ output))
	     (save-excursion
	       (set-buffer buffer)
	       (matlab-shell-help-mode))))))

(defun matlab-shell-run-command (command)
  "Run COMMAND and display result in a buffer.
This command requires an active Matlab shell."
  (interactive (list (read-from-minibuffer
 		      "Matlab command line: "
 		      (cons (matlab-read-line-at-point) 0))))
  (let ((doc (matlab-shell-collect-command-output command)))
    (matlab-output-to-temp-buffer "*Matlab Help*" doc)))

(defun matlab-shell-describe-variable (variable)
  "Get the contents of VARIABLE and display them in a buffer.
This uses the WHOS (Matlab 5) command to find viable commands.
This command requires an active Matlab shell."
  (interactive (list (read-from-minibuffer
 		      "Matlab variable: "
 		      (cons (matlab-read-word-at-point) 0))))
  (let ((doc (matlab-shell-collect-command-output (concat "whos " variable))))
    (matlab-output-to-temp-buffer "*Matlab Help*" doc)))

(defun matlab-shell-describe-command (command)
  "Describe COMMAND textually by fetching it's doc from the Matlab shell.
This uses the lookfor command to find viable commands.
This command requires an active Matlab shell."
  (interactive
   (let ((fn (matlab-function-called-at-point))
	 val)
     (setq val (read-string (if fn
				(format "Describe function (default %s): " fn)
			      "Describe function: ")))
     (if (string= val "") (list fn) (list val))))
  (let ((doc (matlab-shell-collect-command-output (concat "help " command))))
    (matlab-output-to-temp-buffer "*Matlab Help*" doc)))

(defun matlab-shell-apropos (matlabregex)
  "Look for any active commands in MATLAB matching MATLABREGEX.
This uses the lookfor command to find viable commands."
  (interactive (list (read-from-minibuffer
 		      "Matlab command subexpression: "
 		      (cons (matlab-read-word-at-point) 0))))
  (let ((ap (matlab-shell-collect-command-output
	     (concat "lookfor " matlabregex))))
    (matlab-output-to-temp-buffer "*Matlab Apropos*" ap)))
  
(defun matlab-on-prompt-p ()
  "Return t if we Matlab can accept input."
  (save-excursion
    (goto-char (point-max))
    (beginning-of-line)
    (looking-at comint-prompt-regexp)))

(defun matlab-on-empty-prompt-p ()
  "Return t if we Matlab is on an empty prompt."
  (save-excursion
    (goto-char (point-max))
    (beginning-of-line)
    (looking-at (concat comint-prompt-regexp "\\s-*$"))))

(defun matlab-shell-buffer-barf-not-running ()
  "Return a running Matlab buffer iff it is currently active."
  (or (matlab-shell-active-p)
      (error "You need to run the command `matlab-shell' to do that!")))

(defun matlab-shell-collect-command-output (command)
  "If there is a Matlab shell, run the Matlab COMMAND and return it's output.
It's output is returned as a string with no face properties.  The text output
of the command is removed from the Matlab buffer so there will be no
indication that it ran."
  (let ((msbn (matlab-shell-buffer-barf-not-running))
	(pos nil)
	(str nil)
	(lastcmd))
    (save-excursion
      (set-buffer msbn)
      (if (not (matlab-on-prompt-p))
	  (error "Matlab shell must be non-busy to do that"))
      ;; Save the old command
      (goto-char (point-max))
      (beginning-of-line)
      (re-search-forward comint-prompt-regexp)
      (setq lastcmd (buffer-substring (point) (matlab-point-at-eol)))
      (delete-region (point) (matlab-point-at-eol))
      ;; We are done error checking, run the command.
      (setq pos (point))
      (comint-send-string (get-buffer-process (current-buffer))
			  (concat command "\n"))
      (message "Matlab ... Executing command.")
      (goto-char (point-max))
      (while (or (>= pos (point)) (not (matlab-on-empty-prompt-p)))
	(accept-process-output (get-buffer-process (current-buffer)))
	(goto-char (point-max))
	(message "Matlab reading..."))
      (message "Matlab reading...done")
      (save-excursion
	(goto-char pos)
	(beginning-of-line)
	(setq str (buffer-substring-no-properties (save-excursion
						    (goto-char pos)
						    (beginning-of-line)
						    (forward-line 1)
						    (point))
						  (save-excursion
						    (goto-char (point-max))
						    (beginning-of-line)
						    (point))))
	(delete-region pos (point-max)))
      (insert lastcmd))
    str))

(defun matlab-shell-send-string (string)
  "Send STRING to the currently running matlab process."
  (if (not matlab-shell-echoes)
      (let ((proc (get-buffer-process (current-buffer))))
	(goto-char (point-max))
	(insert string)
	(set-marker (process-mark proc) (point))))
  (comint-send-string (get-buffer-process (current-buffer)) string))

(defun matlab-shell-last-error ()
  "In the Matlab interactive buffer, find the last Matlab error, and go there.
To reference old errors, put the cursor just after the error text."
  (interactive)
  (let (eb el)
    (save-excursion
      (end-of-line) ;; In case we are before the linenumber 1998/06/05 16:54sk
      (if (not (re-search-backward gud-matlab-error-regexp nil t))
	  (error "No errors found!"))
      (setq eb (buffer-substring-no-properties
		(match-beginning 2) (match-end 2))
	    el (buffer-substring-no-properties
		(match-beginning 3) (match-end 3))))
    (find-file-other-window eb)
    (goto-line (string-to-int el))))

(defun matlab-shell-dbstop-error ()
  "Stop on errors."
  (interactive)
  (comint-send-string (get-buffer-process (current-buffer))
		      "dbstop if error\n"))

(defun matlab-shell-dbclear-error ()
  "Don't stop on errors."
  (interactive)
  (comint-send-string (get-buffer-process (current-buffer))
		      "dbclear if error\n"))

(defun matlab-shell-demos ()
  "Matlab demos."
  (interactive)
  (comint-send-string (get-buffer-process (current-buffer)) "demo\n"))

(defun matlab-shell-close-figures ()
  "Close any open figures."
  (interactive)
  (comint-send-string (get-buffer-process (current-buffer)) "close all\n"))

(defun matlab-shell-close-current-figure ()
  "Close current figure."
  (interactive)
  (comint-send-string (get-buffer-process (current-buffer)) "delete(gcf)\n"))

(defun matlab-shell-exit ()
  "Exit Matlab shell."
  (interactive)
  (comint-send-string (get-buffer-process (current-buffer)) "exit\n")
  (kill-buffer nil))

;;; matlab-shell based Topic Browser and Help =================================

(defcustom matlab-shell-topic-mode-hook nil
  "*Matlab shell topic hook."
  :group 'matlab-shell
  :type 'hook)

(defvar matlab-shell-topic-current-topic nil
  "The currently viewed topic in a Matlab shell topic buffer.")

(defun matlab-shell-topic-browser ()
  "Create a topic browser by querying an active Matlab shell using HELP.
Maintain state in our topic browser buffer."
  (interactive)
  ;; Reset topic browser if it doesn't exist.
  (if (not (get-buffer "*Matlab Topic*"))
      (setq matlab-shell-topic-current-topic nil))
  (let ((b (get-buffer-create "*Matlab Topic*")))
    (switch-to-buffer b)
    (if (string= matlab-shell-topic-current-topic "")
	nil
      (matlab-shell-topic-mode)
      (matlab-shell-topic-browser-create-contents ""))))

(defvar matlab-shell-topic-mouse-face-keywords
  '(;; These are subtopic fields...
    ("^\\(\\w+/\\w+\\)[ \t]+-" 1 font-lock-reference-face)
    ;; These are functions...
    ("^[ \t]+\\(\\w+\\)[ \t]+-" 1 font-lock-function-name-face)
    ;; Here is a See Also line...
    ("[ \t]+See also "
     ("\\(\\w+\\)\\([,.]\\| and\\|$\\) *" nil nil (1 font-lock-reference-face))))
  "These are keywords we also want to put mouse-faces on.")

(defvar matlab-shell-topic-font-lock-keywords
  (append matlab-shell-topic-mouse-face-keywords
	  '(("^[^:\n]+:$" 0 font-lock-keyword-face)
	    ;; These are subheadings...
	    ("^[ \t]+\\([^.\n]+[a-zA-Z.]\\)$" 1 'underline)
	    ))
  "Keywords useful for highlighting a Matlab TOPIC buffer.")

(defvar matlab-shell-help-font-lock-keywords
  (append matlab-shell-topic-mouse-face-keywords
	  '(;; Function call examples
	    ("[ \t]\\([A-Z]+\\)\\s-*=\\s-*\\([A-Z]+[0-9]*\\)("
	     (1 font-lock-variable-name-face)
	     (2 font-lock-function-name-face))
	    ("[ \t]\\([A-Z]+[0-9]*\\)("
	     (1 font-lock-function-name-face))
	    ;; Parameters: Not very accurate, unfortunately.
	    ("[ \t]\\([A-Z]+[0-9]*\\)("
	     ("'?\\(\\w+\\)'?\\([,)]\\) *" nil nil
	      (1 font-lock-variable-name-face))
	     )
	    ;; Reference uppercase words
	    ("\\<\\([A-Z]+[0-9]*\\)\\>" 1 font-lock-reference-face)))
  "Keywords for regular help buffers.")

;; View-major-mode is an emacs20 thing.  This gives us a small compatibility
;; layer.
(if (not (fboundp 'view-major-mode)) (defalias 'view-major-mode 'view-mode))

(define-derived-mode matlab-shell-help-mode
  view-major-mode "M-Help"
  "Major mode for viewing Matlab help text.
Entry to this mode runs the normal hook `matlab-shell-help-mode-hook'.

Commands:
\\{matlab-shell-help-mode-map}"
  (make-local-variable 'font-lock-defaults)
  (setq font-lock-defaults '((matlab-shell-help-font-lock-keywords)
			     t nil ((?_ . "w"))))
  ;; This makes sure that we really enter font lock since
  ;; kill-all-local-variables is not used by old view-mode.
  (and (boundp 'global-font-lock-mode) global-font-lock-mode
       (not font-lock-mode) (font-lock-mode 1))
  (easy-menu-add matlab-shell-help-mode-menu matlab-shell-help-mode-map)
  (matlab-shell-topic-mouse-highlight-subtopics)
  )

(define-key matlab-shell-help-mode-map [return] 'matlab-shell-topic-choose)
(define-key matlab-shell-help-mode-map "t" 'matlab-shell-topic-browser)
(define-key matlab-shell-help-mode-map "q" 'bury-buffer)
(define-key matlab-shell-help-mode-map
  [(control h) (control m)] matlab-help-map)
(if (string-match "XEmacs" emacs-version)
    (define-key matlab-shell-help-mode-map [button2] 'matlab-shell-topic-click)
  (define-key matlab-shell-help-mode-map [mouse-2] 'matlab-shell-topic-click))

(easy-menu-define
 matlab-shell-help-mode-menu matlab-shell-help-mode-map
 "Matlab shell topic menu"
 '("Matlab Help"
   ["Describe This Command" matlab-shell-topic-choose t]
   "----"
   ["Describe Command" matlab-shell-describe-command t]
   ["Describe Variable" matlab-shell-describe-variable t]
   ["Command Apropos" matlab-shell-apropos t]
   ["Topic Browser" matlab-shell-topic-browser t]
   "----"
   ["Exit" bury-buffer t]))

(define-derived-mode matlab-shell-topic-mode
  matlab-shell-help-mode "M-Topic"
  "Major mode for browsing Matlab HELP topics.
The output of the Matlab command HELP with no parameters creates a listing
of known help topics at a given installation.  This mode parses that listing
and allows selecting a topic and getting more help for it.
Entry to this mode runs the normal hook `matlab-shell-topic-mode-hook'.

Commands:
\\{matlab-shell-topic-mode-map}"
  (setq font-lock-defaults '((matlab-shell-topic-font-lock-keywords)
			     t t ((?_ . "w"))))
  (if (string-match "XEmacs" emacs-version)
      (setq mode-motion-hook 'matlab-shell-topic-highlight-line))
  (easy-menu-add matlab-shell-topic-mode-menu matlab-shell-topic-mode-map)
  )

(easy-menu-define
 matlab-shell-topic-mode-menu matlab-shell-topic-mode-map
 "Matlab shell topic menu"
 '("Matlab Topic"
   ["Select This Topic" matlab-shell-topic-choose t]
   ["Top Level Topics" matlab-shell-topic-browser t]
   "----"
   ["Exit" bury-buffer t]))

(defun matlab-shell-topic-browser-create-contents (subtopic)
  "Fill in a topic browser with the output from SUBTOPIC."
  (toggle-read-only -1)
  (erase-buffer)
  (insert (matlab-shell-collect-command-output (concat "help " subtopic)))
  (goto-char (point-min))
  (forward-line 1)
  (delete-region (point-min) (point))
  (setq matlab-shell-topic-current-topic subtopic)
  (if (not (string-match "XEmacs" emacs-version))
      (matlab-shell-topic-mouse-highlight-subtopics))
  (toggle-read-only 1)
  )

(defun matlab-shell-topic-click (e)
  "Click on an item in a Matlab topic buffer we want more information on.
Must be bound to event E."
  (interactive "e")
  (mouse-set-point e)
  (matlab-shell-topic-choose))

(defun matlab-shell-topic-choose ()
  "Choose the topic to expand on that is under the cursor.
This can fill the topic buffer with new information.  If the topic is a
command, use `matlab-shell-describe-command' instead of changing the topic
buffer."
  (interactive)
  (let ((topic nil) (fun nil) (p (point)))
    (save-excursion
      (beginning-of-line)
      (if (looking-at "^\\w+/\\(\\w+\\)[ \t]+-")
	  (setq topic (match-string 1))
	(if (looking-at "^[ \t]+\\(\\(\\w\\|_\\)+\\)[ \t]+-")
	    (setq fun (match-string 1))
	  (if (and (not (looking-at "^[ \t]+See also"))
		   (not (save-excursion (forward-char -2)
					(looking-at ",$"))))
	      (error "You did not click on a subtopic, function or reference")
	    (goto-char p)
	    (forward-word -1)
	    (if (not (looking-at "\\(\\(\\w\\|_\\)+\\)\\([.,]\\| and\\|\n\\)"))
		(error "You must click on a reference")
	      (setq topic (match-string 1)))))))
    (message "Opening item %s..." (or topic fun))
    (if topic
	(matlab-shell-topic-browser-create-contents (downcase topic))
      (matlab-shell-describe-command fun))
    ))

(defun matlab-shell-topic-mouse-highlight-subtopics ()
  "Put a `mouse-face' on all clickable targets in this buffer."
  (save-excursion
    (let ((el matlab-shell-topic-mouse-face-keywords))
      (while el
	(goto-char (point-min))
	(while (re-search-forward (car (car el)) nil t)
	  (let ((cd (car (cdr (car el)))))
	    (if (numberp cd)
		(put-text-property (match-beginning cd) (match-end cd)
				   'mouse-face 'highlight)
	      (while (re-search-forward (car cd) nil t)
		(put-text-property (match-beginning (car (nth 3 cd)))
				   (match-end (car (nth 3 cd)))
				   'mouse-face 'highlight)))))
	(setq el (cdr el))))))

(defun matlab-shell-topic-highlight-line (event)
  "A value of `mode-motion-hook' which will highlight topics under the mouse.
EVENT is the user mouse event."
  (let* ((buffer (event-buffer event))
	 (point (and buffer (event-point event))))
    (if (and buffer (not (eq buffer mouse-grabbed-buffer)))
	(save-excursion
	  (save-window-excursion
	    (set-buffer buffer)
	    (mode-motion-ensure-extent-ok event)
	    (if (not point)
		(detach-extent mode-motion-extent)
	      (goto-char point)
	      (end-of-line)
	      (setq point (point))
	      (beginning-of-line)
	      (if (or (looking-at "^\\w+/\\(\\w+\\)[ \t]+-")
		      (looking-at "^[ \t]+\\(\\(\\w\\|_\\)+\\)[ \t]+-"))
		  (set-extent-endpoints mode-motion-extent (point) point)
		(detach-extent mode-motion-extent))))))))


;;; M File path stuff =========================================================

(defun matlab-mode-determine-mfile-path ()
  "Create the path in `matlab-mode-install-path'."
  (let ((path (file-name-directory matlab-shell-command)))
    ;; if we don't have a path, find the Matlab executable on our path.
    (if (not path)
	(let ((pl exec-path))
	  (while (and pl (not path))
	    (if (and (file-exists-p (concat (car pl) "/" matlab-shell-command))
		     (not (car (file-attributes (concat (car pl) "/"
							matlab-shell-command)))))
		(setq path (car pl)))
	    (setq pl (cdr pl)))))
    (if (not path)
	nil
      ;; When we find the path, we need to massage it to identify where
      ;; the M files are that we need for our completion lists.
      (if (string-match "/bin$" path)
	  (setq path (substring path 0 (match-beginning 0))))
      ;; Everything stems from toolbox (I think)
      (setq path (concat path "/toolbox/")))
    path))

(defcustom matlab-mode-install-path (list (matlab-mode-determine-mfile-path))
  "Base path pointing to the locations of all the m files used by matlab.
All directories under each element of `matlab-mode-install-path' are
checked, so only top level toolbox directories need be added.
Paths should be added in the order in which they should be searched."
  :group 'matlab-shell
  :type '(repeat (string :tag "Path: ")))

(defun matlab-find-file-under-path (path filename)
  "Return the pathname or nil of PATH under FILENAME."
  (if (file-exists-p (concat path filename))
      (concat path filename)
    (let ((dirs (if (file-directory-p path)
		    ;; Not checking as a directory first fails on XEmacs
		    ;; Stelios Kyriacou <kyriacou@cbmv.jhu.edu>
		    (directory-files path t nil t)))
	  (found nil))
      (while (and dirs (not found))
	(if (and (car (file-attributes (car dirs)))
		 ;; don't redo our path names
		 (not (string-match "/\\.\\.?$" (car dirs)))
		 ;; don't find files in object directories.
		 (not (string-match "@" (car dirs))))
	    (setq found
		  (matlab-find-file-under-path (concat (car dirs) "/")
					       filename)))
	(setq dirs (cdr dirs)))
      found)))

(defun matlab-find-file-on-path (filename)
  "Find FILENAME on the current Matlab path.
The Matlab path is determined by `matlab-mode-install-path' and the
current directory.  You must add user-installed paths into
`matlab-mode-install-path' if you would like to have them included."
  (interactive
   (list
    (let ((default (save-excursion
		     (if (or (bolp)
			     (looking-at "\\s-")
			     (save-excursion (forward-char -1)
					     (looking-at "\\s-")))
			 nil
		       (forward-word -1))
		     (matlab-navigation-syntax
		       (if (looking-at "\\sw+\\>")
			   (match-string 0))))))
      (if default
	  (let ((s (read-string (concat "File (default " default "): "))))
	    (if (string= s "") default s))
	(read-string "File: ")))))
  (if (string= filename "")
      (error "You must specify an M file"))
  (if (not (string-match "\\.m$" filename))
      (setq filename (concat filename ".m")))
  (let ((fname nil)
	(dirs matlab-mode-install-path))
    (if (file-exists-p (concat default-directory filename))
	(setq fname (concat default-directory filename)))
    (while (and (not fname) dirs)
      (if (stringp (car dirs))
	  (progn
	    (message "Searching for %s in %s" filename (car dirs))
	    (setq fname (matlab-find-file-under-path (car dirs) filename))))
      (setq dirs (cdr dirs)))
    (if fname (find-file fname)
      (error "File %s not found on any known paths.  \
Check `matlab-mode-install-path'" filename))))

(defun matlab-find-file-click (e)
  "Find the file clicked on with event E on the current path."
  (interactive "e")
  (mouse-set-point e)
  (let ((f (save-excursion
	     (if (or (bolp)
		     (looking-at "\\s-")
		     (save-excursion (forward-char -1)
				     (looking-at "\\s-")))
		 nil
	       (forward-word -1))
	     (matlab-navigation-syntax
	       (if (looking-at "\\sw+\\>")
		   (match-string 0))))))
    (if (not f) (error "To find an M file, click on a word"))
    (matlab-find-file-on-path f)))


;;; matlab-mode debugging =====================================================

(defun matlab-show-line-info ()
  "Display type and attributes of current line.  Used in debugging."
  (interactive)
  (let ((msg "line-info:")
	(indent (matlab-calculate-indentation (current-indentation)))
	(nexti (matlab-next-line-indentation)))
    (setq msg (concat msg
		      " Line type: " (symbol-name (car indent))
		      " This Line: " (int-to-string (nth 1 indent))
		      " Next Line: " (int-to-string nexti)))
    (if (matlab-lattr-cont)
	(setq msg (concat msg " w/cont")))
    (if (matlab-lattr-comm)
	(setq msg (concat msg " w/comm")))
    (message msg)))

(provide 'matlab)

;;; Change log
;; 18Apr00 by Eric Ludlam <eludlam@mathworks.com>
;;      Finally: Version 2.2.3.
;;
;; 13Apr00 by Eric Ludlam <eludlam@mathworks.com> with original
;;         from Hubert Selhofer <hubert@na.uni-tuebingen.de>
;;      Added `matlab-find-unreachable-code' to detect if (0)
;;        situations.  Use this in font lock to turn that code
;;        into comment colors.
;;      Update `matlab-forward-sexp' to take universal argument.
;;        When set, stop on else/case/catch blocks.  Used by above
;;        fn to not highlight ELSE clause.
;;      Version 2.2.3.b4
;;
;; 06Apr00 by Eric Ludlam <eludlam@mathworks.com>
;;      Updated `matlab-mode-vf-quietafy-buffer' to remove trailing
;;        whitespace too.
;;      Make sure the `fume' vairables exist before modifying them.
;;
;; 08Mar00 by Adrie Dane <A.D.Dane@chem.hull.ac.uk>
;;         tweeked by Eric M. Ludlam
;;      Added support for XEmacs' func-menu program.
;;
;; 19Jan00 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed typo in C-c: binding to uncomment region.
;;
;; 19Jan00 by Hubert Selhofer <hubert@na.uni-tuebingen.de>
;;      Font-lock complex numbers (i,j).
;;
;; 12Jan00 by Eric M. Ludlam <eludlam@mathworks.com> idea by
;;            Jinwei Shen <shenjw@wam.umd.edu>
;;      Added `matlab-show-matlab-shell-buffer' to display the shell
;;        from a matlab buffer.
;;      I moved commands that affect the shell from a text buffer into
;;        their own section.
;;
;; 29Oct99 by Eric M. Ludlam <eludlam@mathworks.com>
;;      `matlab-shell-command-switches' is now a custom string type.
;;      Version 2.2.3.b2
;;
;; 04Aug99 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Customize new font lock keyword list variables.
;;      Fixed `matlab-lattr-middle-block-cont' to return 0 if there IS
;;        a middle block that also contains an END on the same line.
;;
;; 03Aug99 by Hubert Selhofer <hubert@na.uni-tuebingen.de> augmented by
;;            Eric M. Ludlam <eludlam@mathworks.com>
;;      Use Emacs 20's `line-end-position' and `line-beginning-position'.
;;      Use Emacs 20's `regexp-opt' when generating font-lock keywords.
;;      Version 2.2.3.b1
;;
;; 23July99 by Peter J. Acklam <jacklam@math.uio.no>
;;      Font Lock keywords for function defs w/ special EOL.
;;
;; 02July99 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-uncomment-region' command bound to C-c :
;;
;; 04June99 by Ian Wells <I.Wells@tarragon-et.co.uk>
;;      Updated `gud-matlab-error-regexp' to work w/ windows filenames.
;;
;; 20May99 by Eric Ludlam <eludlam@mathworks.com>
;;      Updateded the ftp site where you can get matlab.el
;;
;; 21Apr99 by Eric Ludlam <eludlam@mathworks.com>
;;      Added menu items to toggle code-filling, and periodic details.
;;      Update version to 2.2.2.
;;
;; 14Apr99 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-comment-line-break-function' on M-j for compatibility.
;;
;; 12Apr99 by Eric Ludlam <eludlam@mathworks.com>
;;      In `matlab-valid-end-construct-p' do not use end-of-command.
;;        The end of line command seems to work just fine.
;;
;; 31Mar99 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-end-of-command' to have a better short-cicuit for
;;        long buffers.  (depends on block commands not appearing in matrix)
;;
;; 04Mar99 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-end-of-command' to be savvy of eob.
;;      Fixed `matlab-indent-end-before-ret' to correctly use the
;;        regex functions that need re-indenting backwards.  Use of
;;        `matlab-block-beg-re' was just wrong.
;;
;; 01Mar99 by Eric Ludlam <eludlam@mathworks.com>
;;      Removed ' from menu bindings toggling variables.
;;
;; 10Feb99 by Eric Ludlam <eludlam@mathworks.com>
;;      `matlab-navigation-syntax' will ALSO set `case-fold-search' to nil.
;;
;; 09Feb99 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-beginning-of-command' to not stall inside a
;;        matrix inside a continuation.
;;
;; 29Jan99 by Eric Ludlam <eludlam@mathworks.com>
;;      Added try/catch to hilit19 keywords.
;;
;; 28Jan99 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-indent-end-before-ret' to use `matlab-block-beg-re'
;;        instead of a home-grown regular expression.
;;
;; 17Dec98 by Eric Ludlam <eludlam@mathworks.com>
;;      Updated some documentation.
;;      Post 2.2.1
;;
;; 16Dec98 by Eric Ludlam <eludlam@mathworks.com>
;;      Do not require font lock.  Set up font-lock hooks to generate our
;;        special faces when font lock is loaded (if it isn't already)
;;      Add require 'gud on compile.
;;      Added `matlab-handle-simulink'.  When set to nil, font lock keywords
;;        are modified to remove simulink special stuff.
;;      Fixed `matlab-cursor-in-string' so that the `incomplete' param
;;        works with strings w/ '' in them.
;;      Added celeron to `matlab-mode-vf-quietafy-buffer', and also added
;;        a speed optimization when calling `matlab-end-of-command'.
;;
;; 07Dec98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-maximum-indents' to hold indentation maximum data.
;;      Update `matlab-calculate-indentation' to use the maximums
;;        variable to allow custom indentation levels.  Also fixed so
;;        that if a given paren is immediatly followed by EOL, or if
;;        an indent past arg1's 1st arg is followed by and EOL, then
;;        backup one indentation type.
;;
;; 04Dec98 by Eric Ludlam <eludlam@mathworks.com>
;;      In `gud-matlab-find-file', don't use replace-match on a string.
;;      Fixed nasty bug when a paren appears in a string in an array.
;;      Added `matlab-up-list' which only works backwards at the moment.
;;      Added `array-cont' and `array-end' indentation types.  Array-end types
;;        will line up with the opening bracket, not the array elements.
;;      Hacked several fns to use `array-cont's restrict parameter
;;        to speed things up.
;;
;; 03Dec98 by Eric Ludlam <eludlam@mathworks.com>
;;      Moved simulink keywords into a guadier level.  Added require font-lock.
;;      Fixed `matlab-ltype-help-comm' to not loop endlessly @ bob.
;;      Added `matlab-ispell-strings-region', `matlab-ispell-strings', and
;;        `matlab-ispell-comments'.  Added to menu & keymap.
;;
;; 02Dec98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-show-line-info' to not use old indent functions.
;;      Fixed "proc" ref in `matlab-shell-send-string' which only
;;        worked incidentally.
;;      Release 2.2
;;
;; 15Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      When auto-filling comments at the end of a line, use
;;        auto-fill, and not fill-paragraph.
;;      `matlab-set-comm-fill-prefix' no longer grabs preceding spaces.
;;
;; 12Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      When a regular matlab prompt appears, turn off the overlay arrow.
;;
;; 11Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      Attempted to make `matlab-shell-echos' disply the string about
;;        to be run for save-and-go, and run-region in the fn
;;        `matlab-shell-send-string'
;;
;; 10Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed bug in `gud-matlab-marker-filter' so that instead of
;;        checking for specific prompts, we check specifically for the
;;        error prefix, and display everything else (which includes
;;        the prompt of the INPUT command.
;;
;; 04Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-shell-echos' -> `matlab-shell-echoes'.
;;
;; 04Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      Post 2.1.4d.
;;
;; 03Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-shell-echos' for NT engine program.
;;
;; 02Nov98 by Eric Ludlam <eludlam@mathworks.com>
;;      Due to new engine program for NT, be careful w/ error message in
;;        `matlab-shell'.
;;
;; 31Oct98 by Matt Wette <mwette@alumni.caltech.edu>
;;	changed local variable `comment-column' to value of
;;        `matlab-comment-column' (not the symbol)
;;      changed reference of `matlab-comment-column' to `comment-column'.
;;      Fixed matlab-comment on lines w/ comment.
;;
;; 26Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added help for `matlab-shell' on Windows platform.
;;
;; 23Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed font locking for statement vars for an embedded fn call.
;;      Fixed font locking non-simulink commands starting w/ `sim'
;;
;; 19Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      When auto-filling, ignore lines which are commented regions.
;;
;; 16Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      In `matlab-set-comm-fill-prefix', account for the $$$ of a
;;         region comment.
;;
;; 14Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      When debugging source, convert .p files into .m files before loading.
;;
;; 07Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      A bunch of changes to enable GUD usage under XEmacs 20.4.
;;      Updated some documentation.  Switched to 2.1.4 due to changes.
;;
;; 06Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added binding for `backspace' symbol to not delete prompt.
;;
;; 02Oct98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed matlab shell to beep shell to beep when filtering C-g
;;      Added a command to prevent simple deletion of the prompt in shell mode.
;;
;; 30Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Post 2.1.3i to mailing list.
;;
;; 25Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed a region bug for Emacs 20.2.  Apparently `region-face'
;;         is important, and I shouldn't have used it as a local
;;         variable.
;;
;; 24Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed bug for arg1 functions when 1st arg is a function call,
;;         and the lineup code matches past arg1 of this fn.
;;      Added `matlab-arg1-max-indent-length' to control how long the
;;         first argument must be before we stop indenting.
;;
;; 23Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Made sure region was correctly found for XEmacs users w/ custom.
;;
;; 22Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added optional arg REINDENT to `matlab-insert-end-block' to reindent
;;         the code affected by the new block.
;;
;; 10Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed use of `matlab-indent-past-arg1-functions' so that the `_' char
;;         is also skipped over when backing up.
;;      The font lock keywords for 1st arg is var functions now uses
;;         `matlab-indent-past-arg1-functions' as the default reference.
;;      Added a bunch of simulink keywords to font lock, included a simulink
;;         keyword face, and to `matlab-indent-past-arg1-functions'.
;;
;; 09Sept98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added waitfor in with set/get as a handle/property pair acceptor.
;;      Fixed defcustom rules for for `matlab-unterminated-string-face'
;;         to take the same properties as `font-lock-string-face'.
;;      Added some really gaudy keywords for highlighting interesting
;;         things in a file's help comment.
;;
;; 31Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Post 2.1.3f to mailing list.
;;      Fixed typo in `matlab-shell' relating to keymaps for older emacsen.
;;
;; 21Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-ltype-help-comm' to not hang at beginning of buffer.
;;      Add `matlab-ltype-comm-ignore' returns t on lines which were
;;         hit with `matlab-comment-region'.  `matlab-prev-line'
;;         ignores these when scanning backwards, and
;;         `matlab-calculate-indentation' forces these lines to indent
;;         to column 0.
;;
;; 20Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Update doc for `matlab-case-level' to include cons cell
;;      Update `matlab-calc-indent' and `matlab-next-line-indentation'
;;      to handle cons cell for `matlab-case-level'.
;;
;; 19Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Removed docs about 'nohelpcomment.  This is the default now.
;;
;; 18Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-find-convenient-line-break' to break before a function
;;         name, not instead of at it's opening parenthisis.
;;
;; 14Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-auto-fill' so that when filling many parameters
;;         of strings, the small confusion of string continues doesn't
;;         mess things up (by inserting brackets and things like that.)
;;      Added `matlab-indent-past-arg1-functions' regex which permits
;;         indentation of continuations after certain classes of
;;         functions to occur with the second argument.
;;      Updated `matlab-set-comm-fill-prefix' to be smaller, and smarter.
;;         This eliminated `matlab-set-comm-fill-prefix-post-code'
;;      Removed the unused `matlab-set-code-fill-prefix'.
;;      Fixed `matlab-auto-fill' to fill post code comments first when
;;         applicable.
;;      Fixed `matlab-fill-paragraph' so that if the cursor is AFTER
;;         the ..., it will fill in code mode, not comment mode.
;;         Also, trailing comments are correctly decommented.
;;
;; 12Aug98 by Eric Ludlam <eludlam@mathworks.com>
;;      Rewrote `matlab-fill-region' to go through the region on a
;;         command by command basis calling `matlab-fill-paragraph'
;;      Fixed `matlab-fill-paragraph' to justify code lines, and
;;         insert spaces wher the line breaks were.
;;      Deleted `matlab-justify-comment-line' since we can get that
;;         behavior built in.
;;      Added `matlab-fill-fudge-hard-maximum' to be the hard maximum
;;         limit when filling code.
;;      Fixed `matlab-beginning-of-command' to correctly handle
;;         continued lines at the beginning of a buffer.
;;      Fixed `matlab-calc-indent' to be better about indentation at
;;         the beginning of a buffer.
;;      Fixed `matlab-find-convenient-line-break' to be smarter about
;;         varios potential problems including double-fudge parens,
;;         and before/after space search short-circuiting.
;;
;; 31Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-fill-fudge' which is a fuzzy fill zone.  Also added
;;         `matlab-find-convenient-line-break' which calculates a fuzzy
;;         column to break a line on.  Updated `matlab-do-autofill' to
;;         use the above function.
;;      Updated `matlab-fill-paragraph' to use the new fuzzy line-break
;;         calculation on code.
;;      Added `matlab-ltype-continued-comm' to identify the column on
;;         a previous line where the comment column is.
;;      Updated indentor to indent comments indentically to the previous
;;         line's comment column even if it follows code.
;;      Updated `matlab-beginning-of-command' and `matlab-end-of-command'
;;         to correctly trace to the end of a command's comments.
;;
;; 29Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed `matlab-auto-fill' to prevent filling if typing a space
;;         while the cursor is on the first . or the elipsis.
;;      Fixed `matlab-ltype-endfunction-comm' so it won't barf on end-of-
;;         file conditions.
;;
;; 27Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed first code line in a function indenting problem.  The
;;         variable `matlab-indent-function' was ignored, where code
;;         after a help comment was always indented, and code after
;;         the first function decl, but w/ no help was always not
;;         indented.
;;      Fixed `matlab-beginning-of-command' to correctly handle the
;;         first line in a file containing continuation marks.
;;      Fixed auto fill to not fill lines with elipses on them already.
;;      Fixed `matlab-valid-end-construct-p' to handle unterminated lists
;;         or matrixes.  If a list is unterminated, then all ends are
;;         assumed to be invalid.
;;
;; 20Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Check to make sure `set-keymap-parent' exists before using it.
;;
;; 07Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Moved `matlab-mode-verify-fix-functions' to custom variable section,
;;         and turned it into a defcustom declaration.
;;      Added `matlab-lattr-array-cont' function to identify lines that
;;         have unstarted/ended arrays in them.
;;      `matlab-beginning-of-command' and `matlab-end-of-command' now track
;;         across unbounded array constants.
;;
;; 06Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Put `matlab-indent-function' back to nil.
;;      `matlab-mode-vf-quietafy-buffer' can take optional ignored FAST arg.
;;      Release 2.1.2.
;;
;; 02Jul98 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-mode-vf-quietafy-buffer' which finds lines which may
;;         produce ANS style output.
;;      When `matlab-indent-function' is 'nohelpcomment, then the first
;;         comment in a function is unindented while the function body
;;         is indented.
;;
;; 22Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Even more string highlighting fixes for spaces in strings.
;;
;; 17Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Re-worked the string/comment font locking to use functions instead
;;         of string matching.  The functions are now much more accurate.
;;
;; 16Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed auto-fill prefix for comments to not require a space.
;;
;; 15Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      More string font lock fixing comments directly after a string.
;;
;; 10Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed unterminated strings that contain comment characters,
;;         and don't highlight the comments in unterminated strings.
;;      Discovered why XEmacs 20 wouldn't highlight some items, and
;;         fixed it.  Required lots of 'appends, plus the removal
;;         of 'prepend for comments which allowed other keywords to show
;;         through.
;;
;; 10Jun98 by Peter J. Acklam <jacklam@math.uio.no>
;;      Fixed unterminated strings with quoted quotes.
;;
;; 05Jun98 by Stelios Kyriacou <kyriacou@cbmv.jhu.edu>
;;      Update `gud-matlab-error-regexp' to handle warnings.
;;      Update `matlab-shell-last-error' to find errors the cursor is in.
;;
;; 04Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Attempt to make `matlab-unterminated-string-face' do the defface thing.
;;
;; 04Jun98 by Stelios Kyriacou <kyriacou@cbmv.jhu.edu>
;;      Fixed `matlab-find-file-on-path' to correctly search the list of paths.
;;         Added check that a path is a directory before fetching it's files.
;;      Fixed a bug when I merged a change in from jacklam which I seem to
;;         have forgotten to put into this history.
;;
;; 03Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      `matlab-unterminated-string-face' is now a self-referencing variable.
;;      Post version 2.1.1
;;
;; 02Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed the function `matlab-mode-determine-mfile-path' to not fail.
;;      Updated `matlab-find-file-on-path' to handle nil's in the list
;;         and provide helpful errors
;;
;; 01Jun98 by Eric Ludlam <eludlam@mathworks.com>
;;      Post version 2.1
;;
;; 27May98 by Eric Ludlam <eludlam@mathworks.com>
;;      Enabled `matlab-mode-determine-mfile-path' and used it to
;;        define the variable `matlab-mode-install-path'.  This is
;;        then used by the new commands `matlab-find-file-on-path' and
;;        `matlab-find-file-click'  Added these to the keymap and meny.
;;
;; 22May98 by Dan Nicolaescu <done@wynken.ece.arizona.edu>
;;      Fixed derived modes to correctly font lock upon creation.
;;
;; 19May98 by Peter J. Acklam <jacklam@math.uio.no>
;;      New function highlighting regexps which are more accurate.
;;
;; 11May98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Ran new checkdoc on the file and fixed all calls to `error'
;;
;; 11May98 by Peter J. Acklam <jacklam@math.uio.no>
;;      Fixed a string highlighting bug.
;;
;; 11May98 Michael Granzow <mg@medi.physik.uni-oldenburg.de>
;;      Found bug in `matlab-keywords-boolean'.
;;
;; 08May98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      CR after unterminated END will error, but still insert the CR.
;;
;; 08May98 by Hubert Selhofer <hubert@na.mathematik.uni-tuebingen.de>
;;      CR when (point) == (point-min) no longer errors
;;
;; 05May98 by Hubert Selhofer <hubert@na.mathematik.uni-tuebingen.de>
;;      Many spelling fixes in comments, and doc strings.
;;      Adjusted some font-lock keywords to be more compact/effecient.
;;
;; 30Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      %endfunction unindenting can now have arbitrary text after it.
;;
;; 24Apr98 by Peter J. Acklam <jacklam@math.uio.no>
;;      Fixed highlighting of for statements w/ traling comments.
;;
;; 23Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed -vf-block functions to have more restrictive before-keyword
;;        so we don't accidentally match keywords at the end of symbols.
;;
;; 22Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Release 2.0 to web site and newsgroups.
;;      Ran checkdoc/ispell on entire file.
;;      Cleaned up some compile-time warnings.
;;      Verified XEmacs compatibility.
;;
;; 13Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed bug in `matlab-mode-vf-functionname' to prevent infinite loop
;;        on empty files.
;;
;; 10Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Added break to highlighted keywords.
;;      Case variable highlighting now stops at comment endings.
;;
;; 07Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      `matlab-ltype-comm' no longer demands a space after the %.
;;      Indentor now unindents the comment %endfunction.
;;      Removed transposing transpose part.  It broke quoted quotes.
;;
;; 02Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Comments appearing at the end of a function, and just before a new
;;          subfunction, are now unintented if `matlab-indent-function' is
;;          non-nil.  This lets matlab users use %endfunction at the end
;;          of a function, and get the indentation right.
;;
;; 01Apr98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Smarter font lock for case (jacklam@math.uio.no)
;;      Auto fill accounts for chars inserted based on the variable
;;          `matlab-fill-count-ellipsis-flag'.
;;      Auto fill will now fill a string by putting it into brackets
;;          controlled by `matlab-fill-strings-flag'.
;;
;; 18Mar98 by Peter J. Acklam <jacklam@math.uio.no>
;;      Enabled multi-line function definitions in font-lock and imenu.
;;
;; 16Mar98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed potential error in comment searching around ...
;;      Fixed many function regexp's as per Peter J. Acklam's
;;          <jacklam@math.uio.no> suggestion.
;;
;; 09Mar98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed `tempo-template-matlab-function' to work correctly.
;;      Fixed indentation for many other templates.
;;      Made sure the verifier uses navigation syntax.
;;
;; 23Feb98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed problem with x='%'    % ' this shouldn't work
;;      Fixed a problem w/ strung up brackets messing up valid
;;         end identification.
;;
;; 17Feb98 by Aki Vehtari <Aki.Vehtari@hut.fi>
;;      Fixed prompt regexp to include the debugging K.
;;
;; 11Feb98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Made `matlab-mode-vf-functionname' more robust to arbitrary
;;         versions of a function definition.  This includes allowing
;;         comments and blank lines before the first fn definition.
;;      Fixed up the font lock keywords for functions some
;;
;; 10Feb98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed problem with derived view mode.
;;      Fixed font locking of globals to allow a ; at the end.
;;      Fixed function name verifier to not allow = on next line.
;;         It used to match invalid expressions.
;;      `matlab-shell-collect-command-output' now uses a different prompt
;;         detector when waiting for output.  This prevents early exit.
;;
;; 09Feb98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Updated `matlab-indent-line' to not edit the buffer if no changes
;;         are needed, and to make after cursor position smarter.
;;
;; 05Feb98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Added completion semantics and lists for HandleGraphics property lists
;;      Added `matlab-completion-technique' and made it's default value
;;         'completion.  This shows a buffer of completions instead of
;;         cycling through them as the hippie-expand command does.
;;
;; 26Jan98 by Aki Vehtari <Aki.Vehtari@hut.fi>
;;      The Matlab logo variable now uses XEmacs 20.4 locate function.
;;      Small cleanups
;;
;; 26Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Updated `matlab-fill-paragraph' to use a better fill prefix.
;;      Moved code sections around, and added page breaks for navigation.
;;
;; 23Jan98 by Aki Vehtari <Aki.Vehtari@hut.fi>
;;	(matlab-frame-init): Fix typo in menu.
;;	(matlab-output-to-temp-buffer): Use matlab-shell-help-mode.
;;	(matlab-shell-run-region): New function.
;;	(matlab-shell-collect-command-output): Remove (goto-char (point-max)).
;;	(matlab-shell-topic-mode-hook): Name change.
;;	(matlab-shell-topic-browser): Use matlab-shell-topic-mode.
;;	(matlab-shell-help-mode): New mode. Derive from view-major-mode.
;;	(matlab-shell-help-mode-menu): Define.
;;	(matlab-shell-topic-mode): Name change and derive from
;;	   matlab-shell-help-mode.
;;	(matlab-shell-topic-mode-menu): Name change.
;;
;; 22Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Make `matlab-comment' insert `matlab-comment-s' on lines with
;;         no text when there there is no previous comment line to mimic.
;;
;; 21Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Fixed a few templates.  Added `matlab-if-else'.
;;      `matlab-insert-end-block' will now add a comment consisting of
;;         the text starting the block being ended.
;;      Added colors to variables defined with the global command.
;;      Added `matlab-complete-symbol' which uses `matlab-find-recent-variable'
;;         which searches backwards for variables names, and
;;         `matlab-find-user-functions' which finds user functions.
;;         There are also `matlab-*-completions' for solo commands
;;         (if, else, etc), value commands, and boolean commands.
;;         The current semantic state is found w/ `matlab-lattr-semantics'
;;
;; 20Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;     Changed `matlab-block-scan-re' to have a limiting expression at
;;         the beginning.  This makes sexp scanning faster by skipping
;;         more semantically bad matches.
;;     Forward/backward sexp now watch `matlab-scan-on-screen-only', which
;;         make them stop when the scan falls off the screen.  Useful for
;;         making the block highlighter *much* faster for large constructs,
;;         and is logical since we can't see the highlight anyway.
;;     Added `matlab-block-verify-max-buffer-size' to turn off long checks
;;         on big buffers during save only.  Requesting a verify will do
;;         the checks anyway.
;;     Fixed block verifiers to check that found end keywords are also
;;         valid block terminators.
;;
;; 19Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;     Fixed `gud-matlab-marker-filter' and `matlab-join-comment-lines'
;;         to not use `replace-match's fifth argument.
;      Replaced `matlab-insert-' with tempo templates where appropriate.
;;
;; 19Jan98 by Aki Vehtari <Aki.Vehtari@hut.fi>
;;      Fixed `matlab-mode-vf-functionname' to use a correct form
;;         of `replace-match' for XEmacs.
;;      Suggested form of `matlab-navigation-syntax'.
;;
;; 14Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Added manu `matlab-insert-' functions, including:
;;        `switch-block', `next-case', `end-block', `if-block',
;;        `for-block', `try-block', `while-block'.
;;      Added `matlab-stringify-region' which takes a region, and
;;        converts it to a string by adding ' around it, and quoting
;;        all the quotes in the region.
;;      Added an insertion prefix C-c C-c for all insert commands, and
;;        the stringify function.
;;      `matlab-auto-fill' is now assigned to `normal-auto-fill-function',
;;        which is an Emacs 20 thing for auto-fill minor mode.
;;      Added `matlab-beginning-of-command' and `end-of-command' which
;;        moves across lines w/ continuation.
;;      Changed `matlab-lattr-cont' to allow continuation on lines
;;        ending in semicolon.  Is this correct?
;;      Changed the main menu to have submenues for navigation,
;;        formatting, and the new insert functions.
;;      Fixed `matlab-forward-sexp' to not skip over brackets which
;;        was appeared to be a missunderstanding.
;;      Block highlighter and block verifiers no longer treat function
;;        as requiring an "end" keyword.
;;
;; 09Jan98 by Eric M. Ludlam <eludlam@mathworks.com>
;;      Based on code donated by Driscoll Tobin A <tad@cauchy.colorado.edu>
;;        `matlab-fill-paragraph' designed for M file help text, which
;;        will fill/justify comment text, and uses paragraph rules.
;;        `matlab-fill-comment-line' does not know about paragraphs.
;;      `matlab-cursor-in-string' can now take an optional argument
;;        which will identify an unterminated string.
;;      `matlab-auto-fill' will not fill strings, and if the string is
;;        not yet terminated, will also not fill it.  When the string
;;        is terminated, the split will happen after the string, even
;;        if it occurs after the `fill-column'.
;;
;; 08Jan98 by Aki Vehtari  <Aki.Vehtari@hut.fi>
;;      XEmacs compatibility associated with timers.
;;      XEmacs optimizations associated with point-at-[eb]ol.
;;      Turned key sequences from strings to Emacs/XEmacs wide [()] form
;;      Documentation string fixes.
;;      Customizable hooks.  Also update other custom vars.
;;      Remove `matlab-reset-vars' and turn variables controlled by
;;        `matlab-indent-function' into functions.
;;      Some menu re-arrangements & topic-browser menu.
;;      Use matlab-region-face instead of 'region when highlighting stuff.
;;      `matlab-shell-exit' now deletes the buffer when it's done.
;;      `write-contents-hooks' is forced buffer local.
;;      Fixed `matlab-output-to-temp-buffer'.
;;      Made matlab-shell group.
;;
;; 07Jan98 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed indenting problem when end is first used as matrix index
;;        and is also the first word on a line.
;;
;; 07Jan98 by Aki Vehtari  <Aki.Vehtari@hut.fi>
;;      Fixed comments to use add-hook instead of setq.
;;      Variable name cleanup.  Added ###autoload tags to -mode and -shell.
;;      Removed some unused variables.
;;
;; 24Dec97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added `matlab-shell-enable-gud-flag' to control if the GUD features
;;        are used in shell mode or not.  This is automatically set to nil
;;        when certain GUD features are not present
;;      Added stop/clear if error to menu to help people out w/ the debugger.
;;      Added block highlighting of if/for/etc/end constructs.
;;      Fixed up cursor-in-string even more to handle bol better.
;;      Fixed problem w/ syntax table installing itself in funny places
;;        and fixed the fact that tab was now treated as whitespace.
;;
;; 22Dec97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added verify/fix mode when saving.  Added function name check.
;;        Added unterminated block check.  Added unmatched end check.
;;      Fixed `matlab-backward-sexp' to error on mismatched end/begin blocks.
;;
;; 15Dec97 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed some string stuff, and added checks when starting the shell.
;;
;; 10Dec97 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed string font-locking based on suggestions from:
;;        Hubert Selhofer <hubert@na.uni-tuebingen.de>
;;        Peter John Acklam <jacklam@math.uio.no>
;;        Tim Toolan <toolan@ele.uri.edu>
;;      Fixed comment with ... indenting next line.
;;      Made command output collecting much faster.
;;
;; 10Dec97 merged the following:
;; 21May97 by Alf-Ivar Holm <alfh@ifi.uio.no>
;;      Added smart initial values of matlab help commands.
;;      Running commands in matlab-shell remembers old cmd line
;;      Commands can be run when a parial command line is waiting
;;      Changed apropo to apropos where applicable.
;;
;; 9Dec98 merged the following:
;; 30May97 by Hubert Selhofer <hubert@na.uni-tuebingen.de>
;;      Added 'endfunction' to keyword patterns (octave), slightly
;;          changed regexp for better performance.
;;      Added 'endfunction' to `matlab-block-end-pre-no-if' for compliance
;;          with octave.
;;      Fixed `matlab-clear-vars' (symbol names were incorrectly
;;           spelled matlab-matlab-*).
;;      Fixed typo in `matlab-really-gaudy-font-lock-keywords'.
;;
;; 26Nov97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added support for cell array indenting/continuation.
;;      Begin re-enumeration to V 2.0
;;
;; 11Nov97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added custom support for [X]emacs 20.
;;
;; 11Nov97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added beginning/end-defun navigation functions.
;;      Ran through latest version of checkdoc for good doc strings.
;;
;; 04Sep97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added try/catch blocks which are new Matlab 5.2 keywords.
;;
;; 02Sep97 by Eric Ludlam <eludlam@mathworks.com>
;;      Made auto-fill mode more robust with regard to comments
;;        at the end of source lines
;;
;; 13Aug97 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed indentation bugs regarding the beginning of buffer.
;;      Added GUD support into matlab-shell.  Debugger commands will
;;        now automatically check the stack and post the files being
;;        examined via this facility.
;;
;; 26Jun97 by Eric Ludlam <eludlam@mathworks.com>
;;      Help/Apropo buffers are now in Topic mode, and are highlighted.
;;        This allows navigation via key-clicks through the help.
;;      Describe-command can find a default in the current M file.
;;      Mouse-face set to make clickable items mouse-sensitive in topic buffers
;;
;; 25Jun97 by Anders Stenman <stenman@isy.liu.se>
;;      Some XEmacs hacks. Implemented highlighting of subtopics and
;;      commands under mouse in topic-browser mode. Added a nice Matlab
;;      logo in matlab-shell mode.
;;      See: http://www.control.isy.liu.se/~stenman/matlab
;;
;; 13Jun97 by Anders Stenman <stenman@isy.liu.se>
;;      Use the easymenu package for menus. Works both in XEmacs and
;;      FSF Emacs. Bound TAB to comint-dynamic-complete-filename in
;;      matlab-shell mode. Added a function matlab-shell-process-filter
;;      to filter out some escape character rubbish from Matlab output.
;;
;; 20May97 by Matt Wette <mwette@alumni.caltech.edu>
;;	Released as version 1.10.0.
;;
;; 16May97 by Eric Ludlam <eludlam@mathworks.com>
;;      Ran through checkdoc to fix documentation strings.
;;
;; 15May97 by Matt Wette <mwette@alumni.caltech.edu>
;;	Added shell-mode-map bindings; run matlab-shell-mode-hook, not
;;	matlab-shell-mode-hooks (PMiller). Changed keymaps for \C-<letter>,
;;	which conflicted w/ emacs style guidelines.
;;
;; 08May97 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed forward/backward sexp error when end keyword appears as
;;            word component such as the symbol the_end
;;
;; 22Apr97 by Eric Ludlam <eludlam@mathworks.com>
;;      Fixed comment where `indent-function' was incorrectly spelled
;;      Fixed indentation when strings contained [] characters.
;;      Fixed indentation for multi-function files
;;      Added Imenu keywords.  Permits use w/ imenu and emacs/speedbar
;;      The actual version of matlab file is not in a variable
;;      Keybinding for forward/backward sexp
;;      New function finds the mfile path.  Not used for anything useful yet.
;;      Added matlab-shell/emacs io scripting functions.  Used this in
;;            a topic/help/apropo browser.  Could be used w/ other
;;            functions quite easily.
;;
;; 12Mar97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added new `matlab-shell-collect-command-output' to use for running
;;            matlab commands and getting strings back.  Used this function
;;            to create `-describe-function', `-describe-variable', and
;;            `-apropo'.  Should be useful for other things too.
;;      Added some XEmacs specific stuff.
;;
;; 07Mar97 by Matt Wette <mwette@alumni.caltech.edu>
;;	Fixed a few xemacs problems.  Released as 1.09.0.
;;
;; 03Mar97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added expressions to handle blocks which are not terminated with
;;            the 'end' command
;;      Added `matlab-shell-save-and-go' function to automatically run
;;            a function after saving it.
;;      Bug fixes to `matlab-forward-sexp'
;;      Improved font lock interface to take advantage of the user
;;            variable `font-lock-use-maximal-decoration'
;;
;; 24Feb97 by Eric Ludlam <eludlam@mathworks.com>
;;      Added more font locking, plus font locking of `matlab-shell'
;;      Added `matlab-backward-sexp',`matlab-cursor-in-string-or-comment'
;;      Added ability to indent switch/case/case/otherwise/end blocks
;;            as per manual specifications for matlab v5.0
;;      Added command for matlab-shell to goto the last reported error
;;      Modified matlab-shell to use comint features instead of hand
;;            crafted workarounds of the defaults
;;
;; 07Dec96 by Matt Wette <mwette@alumni.caltech.edu>
;;	incorporated many fixes from Mats Bengtsson <matsb@s3.kth.se>;
;;	font-lock comment/string fixes, Eric Ludlam <eludlam@mathworks.com>;
;;	added support for switch construct;
;;
;; 01Aug96 by Matt Wette <mwette@alumni.caltech.edu>
;;	fixed to jive w/ emacs lib conventions: changed name of file from
;;	matlab-mode.el to matlab.el (14 char limit); released as 1.08.0
;;
;; 28Apr96 by Matt Wette <mwette@alumni.caltech.edu>
;;	comments lines w/ just % are now hilighted; syntax table: "-2" changed
;;	to " 2"; released 1.07.6
;;
;; 30Jan96 by Matt Wette <mwette@alumni.caltech.edu>
;;	fixed problem w/ emacs-19.30 filling and auto-fill problem thanks to
;;	Mats Bengtsson <matsb@s3.kth.se>; started implementation of matlab-
;;	shell, based on comint and shell-mode; released 1.07.5
;;
;; 25Jan96 by Matt Wette <mwette@alumni.caltech.edu>
;;	added "global" to font-lock, hilit keywords; fixed indenting of 2nd
;;	line if first ends in ...; filling is broken for FSF19.30 (works for
;;	FSF19.28); torkel fixes to matlab-reset-vars; fixed indent bug
;;	reported by Trevor Cooper;
;;
;; 20Jan96 by Matt Wette <mwette@alumni.caltech.edu>
;;	cleaned up commenting; added preliminary `matlab-shell' mode,
;;	rel 1.07.4
;;
;; 19Jan96 by Matt Wette <mwette@alumni.caltech.edu>
;;	commented out `debug-on-error'; got hilit to work for sam
;;
;; 18Jan96 by Matt Wette <mwette@alumni.caltech.edu>
;;	fixed problem int `matlab-prev-line' which caused fatal `matlab-mode';
;;	crash fixed problem with indenting when keywords in comments; still
;;	haven't cleaned up comment formatting ...
;;
;; 21Jul95 by Matt Wette <mwette@alumni.caltech.edu>
;;	fixes by Bjorn Torkelsson <torkel@cs.umu.se>: replaced
;;	lattr-comment w/ lattr-comm to fix inconsistency; added
;;	function to font-lock keywords, added function name to
;;	`font-lock-function-name-face'.  He had also added function as
;;	a block begin keyword.  This should be an option since it will
;;	cause the body of a function to be indented.  Worked on
;;	filling.  More work on filling.  fixed many bugs reported by
;;	Rob Cunningham.  Pulled cadr.
;;
;; 13Jul95 by Matt Wette <mwette@mr-ed.jpl.nasa.gov>
;; 	changed indenting for continuation lines in calc-deltas to use
;;	cont-level; changed syntax-table;  changed the way the return key is
;;	mapped; released version 1.07.1
;;
;; 08Jul95 by Matt Wette <mwette@mr-ed.jpl.nasa.gov>
;;	This is a fairly major rewrite of the indenting functions to
;;	fix long- startednding problems arising from keywords and
;;	percents in strings.  We may have to add more heuristics later
;;	but this may work better.  Changed comment region string.
;;	Released version 1.07.0.
;;
;; 10Oct94 by Matt Wette <mwette@csi.jpl.nasa.gov>
;;	changed 'auto-fill-mode' to `auto-fill-function'; changed
;;	`comment-indent-' to `comment-indent-function'; fixed percents
;;	in strings being interpreted as comments, but a % for comment
;;	should not be followed by [disx%]
;;
;; 23Nov93 by Matt Wette <mwette@csi.jpl.nasa.gov>
;;	added Lucid emacs, GNU emacs font-lock and lhilit support; repaired
;;	mtlb-block-{beg,end}-kw (Thanks to Dave Mellinger <dkm1@cornell.edu>)
;;	removed string delim entry from matlab-mode-syntax-table (MATLAB lang
;;	sucks here -- why not use " for strings?).   Released vers 1.06.0
;;
;; 10Aug93 by Matt Wette <mwette@csi.jpl.nasa.gov>
;;	added `matlab-indent-end-before-return'; indent may be fixed now
;;	still not working for emacs 19
;;
;; 02Aug93 by Matt Wette <mwette@csi.jpl.nasa.gov>
;;	fixed error in `mtlb-calc-indent'; bumped version to 1.05.1;
;;	added `mtlb-prev-line'; bumped version to 1.05.3; added
;;	`mtlb-calc-block-indent'
;;
;; 01Aug93 by Matt Wette <mwette@csi.jpl.nasa.gov>
;;	Fixed bug which treated form as block-begin keyword.  Reworked
;;	`mtlb-calc-indent' -- seems to work better w/ redundant cont
;;	lines now. Bumbed version to 1.05.
;;
;; 13Jun93 by Matt Wette <mwette@csi.jpl.nasa.gov>
;;	Changed `linea' to `lattr', `linet' to `ltype', fixed
;;	Bumped version number from 1.03bb to 1.04.
;;
;; 02May91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Added `matlab-auto-fill' for `auto-fill-hook' so that this
;;	mode doesn't try to fill matlab code, just comments.
;;
;; 22Apr91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Changed `mtlb-ltype-cont' to `mtlb-lattr-cont',
;;	`mtlb-ltype-comment-on-line' to `mtlb-lattr-comment' and
;;	`mtlb-ltype-unbal-mexp' to `mtlb- attr-unbal-mext' to
;;	emphasize that these are line attributes and not line types.
;;	Modified `matlab-line-type' to reflect the change ini logic.
;;
;; 18Apr91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;      Modified `matlab-comment-return' so that when hit on a line with a
;;	comment at the end it will go to the comment column.  To get the
;;	comment indented with the code, just hit TAB.
;;
;; 17Apr91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Received critique from gray@scr.slb.com.  Changed ml- to mtlb-
;;	due to possible conflict with mlsupport.el routines.  Added
;;	`matlab-comment' -line-s and -on-line-s.  Fixed bug in
;;	`matlab-comment' (set-fill-prefix).  `matlab-comment-return'
;;	now works if called on a non-comment line.
;;
;; 04Mar91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Added const `matlab-indent-before-return'.  Released Version 1.02.
;;
;; 02Feb91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Changed names of `ml-*-line' to `ml-ltype-*'.  Cleaned up a
;;	lot. Added `ml-format-comment-line', fixed `ml-format-region'.
;;	Changed added "-s" on end of `matlab-comment-region' string.
;;	Justify needs to be cleaned up.
;;
;; Fri Feb  1 09:03:09 1991; gray@scr.slb.com
;;      Add function `matlab-comment-region', which inserts the string
;;      contained in the variable matlab-comment-region at the start
;;      of every line in the region.  With an argument the region is
;;      uncommented.  [Straight copy from fortran.el]
;;
;; 25Jan91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Got indentation of matrix expression to work, I think.  Also,
;;	added tabs to comment start regular-expression.
;;
;; 14Jan91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;	Added functions `ml-unbal-matexp' `ml-matexp-indent' for matrix
;;	expressions.
;;
;; 07Jan91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;      Many changes.  Seems to work reasonably well.  Still would like
;;      to add some support for filling in comments and handle continued
;;      matrix expressions.  Released as Version 1.0.
;;
;; 04Jan91 by Matt Wette, mwette@csi.jpl.nasa.gov
;;      Created.  Used eiffel.el as a guide.

;;; matlab.el ends here
