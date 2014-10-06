#!/bin/bash

dirlist=
target=
force=
do_export=0

while (( $# ))
do
	case "$1" in
		-speedup|-debug|-conv)
			target="$target ${1#-}";;
		-all) target=all;;
		-force) force=1;;
		-noexport) do_export=0;;
		-skip) shift ;
			if (( !$# )) || isint $1 ; then skip=$1 ; fi
		;;
		*)	dirlist="$dirlist $1";;
	esac
	shift
done

isint() { test "$1" && printf '%d' "$1" >/dev/null 2>&1; }

function export_execs \
{
	# args = origin_dir name target(s)
	for target in ${@:3}
	do
		echo "Exporting from $1 to $2 (target=$target)"
		if [[ "$target" == "all" || "${3/debug}" != "$3" ]]
		then
			cp $1/cg ~/debug_execs/$2_ompss
			cp $1/cg_seq_cg ~/debug_execs/$2_plain
		fi

		if [[ "$target" == "all" || "${3/conv}" != "$3" ]]
		then
			cp $1/cg_conv ~/conv_execs/$2_ompss
			cp $1/cg_seq_conv ~/conv_execs/$2_plain
		fi

		if [[ "$target" == "all" || "${3/speedup}" != "$3" ]]
		then
			cp $1/cg_speedup ~/speedup_execs/$2_ompss
			cp $1/cg_seq_speedup ~/speedup_execs/$2_plain
		fi
	done
}

if [[ "$target" = "" ]] ; then target=all ; fi
fail=0

while read l
do
	if (( $skip )) ; then let skip="$skip - 1"; continue ; fi
	
	IFS=:
	item=($l)
	IFS=' '

	n=${item[0]}
	dir=${item[1]}
	opt=${item[2]}
	dest=${item[3]}

	echo make $opt $target
	make $opt $target &> /tmp/make_out
	ret=$?

	# do export

	if (( $ret )) ; then 
		echo 
		echo "Failed with #$n $dir : \"$opt\""
		echo see error messages in /tmp/make_out
		fail=$n
		break
	elif [[ "$dest" != "" && $do_export -ne 0 ]] ; then
		export_execs "$dir" "$dest" "$target"
	elif [[ $do_export -ne 0 ]] ; then
		echo "No exports to be made for $dir"
	fi
done < <(cat -n configs.txt | sed "s/^ *\([0-9]\+\)\s\+/\1:/")

if [[ $fail -eq 0 ]]; then echo compiled all configurations successefully ! ; fi
exit $fail


