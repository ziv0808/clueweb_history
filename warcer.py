import os
import ast
import uuid
import unicodedata
import pandas as pd

import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


def build_interval_list(
        work_year,
        frequency):

    interval_list = []
    for i in range(1, 13):
        if frequency == '2W':
            interval_list.extend(
                [work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-01',
                 work_year + "-" + (2 - len(str(i))) * '0' + str(i) + '-16'])
        else:
            raise Exception('build_interval_dict: Unknoen frequency...')

    return interval_list

def parse_timestamp(timestamp_str):
    month_abrev_dict = {
        '01' : 'Jan',
        '02' : 'Feb',
        '03' : 'Mar',
        '04' : 'Apr',
        '05' : 'May',
        '06' : 'Jun',
        '07' : 'Jul',
        '08' : 'Aug',
        '09' : 'Sep',
        '10' : 'Oct',
        '11' : 'Nov',
        '12' : 'Dec'}
    weekday_abrev_dict = {
        0: 'Mon',
        1: 'Tue',
        2: 'Wed',
        3: 'Thu',
        4: 'Fri',
        5: 'Sat',
        6: 'Sun'}

    date_datetime = pd.to_datetime(timestamp_str)
    date_splitted = timestamp_str.split(' ')[0].split('-')
    date_parsed = weekday_abrev_dict[date_datetime.weekday()] + ', ' + date_splitted[2] + ' ' + month_abrev_dict[date_splitted[1]] + ' ' +\
                    date_splitted[0] + " " + timestamp_str.split(' ')[1] + ' GMT'
    return date_parsed


def create_warc_head(
        warc_date,
        warc_info_id):
    warc_info_str = "WARC/0.18\n" +\
                    "WARC-Type: warcinfo\n" +\
                    "WARC-Date: " + warc_date + "\n" +\
                    "WARC-Record-ID: <urn:uuid:" + warc_info_id + ">" + "\n" + \
                    "Content-Type: application/warc-fields\n" +\
                    "Content-Length: "
    next_str      = "\n\nsoftware: Nutch 1.0-dev (modified for clueweb09)\n" +\
                    "isPartOf: clueweb09-en\n" +\
                    "description: clueweb09 crawl with WARC output\n" +\
                    "format: WARC file version 0.18\n" + \
                    "conformsTo: http://www.archive.org/documents/WarcFileFormat-0.18.html\n\n"
    return warc_info_str + str(len((next_str).encode('utf-8')) - 3) + next_str


def create_warc_record(
        docno,
        timstamp,
        url,
        html,
        warc_info_id,
        warc_date):
    # normalize unicode form
    html = unicodedata.normalize("NFKD", html)

    record_str = "WARC/0.18\n" +\
                 "WARC-Type: response\n" +\
                 "WARC-Target-URI: " + url + "\n" +\
                 "WARC-Warcinfo-ID: " + warc_info_id + "\n" + \
                 "WARC-Date: " + warc_date + "\n" + \
                 "WARC-Record-ID: <urn:uuid:" + str(uuid.uuid1())  + ">" + "\n" + \
                 "WARC-TREC-ID: " + docno + "\n" + \
                 "Content-Type: application/http;msgtype=response" + "\n" + \
                 "WARC-Identified-Payload-Type: " + "\n" + \
                 "Content-Length: "
    try:
        next_str   = "HTTP/1.1 200 OK" + "\n" + \
                     "Content-Type: text/html" + "\n" + \
                     "Date: " + parse_timestamp(timstamp) + "\n" + \
                     "Pragma: no-cache"+ "\n" + \
                     "Cache-Control: no-cache, must-revalidate" + "\n" + \
                     "X-Powered-By: PHP/4.4.8" + "\n" + \
                     "Server: WebServerX" + "\n" + \
                     "Connection: close" + "\n" + \
                     "Last-Modified: " + parse_timestamp(timstamp) + "\n" + \
                     "Expires: Mon, 20 Dec 1998 01:00:00 GMT" + "\n" + \
                     "Content-Length: " + str(len((html).encode('utf-8')) + 1) + "\n\n" + html + '\n\n'

        record_str += str(len((next_str).encode('utf-8')) + 1) + "\n\n" + next_str
    # except Exception as e:
    #     try:
    #         next_str = "HTTP/1.1 200 OK" + "\n" + \
    #                "Content-Type: text/html" + "\n" + \
    #                "Date: " + parse_timestamp(timstamp) + "\n" + \
    #                "Pragma: no-cache" + "\n" + \
    #                "Cache-Control: no-cache, must-revalidate" + "\n" + \
    #                "X-Powered-By: PHP/4.4.8" + "\n" + \
    #                "Server: WebServerX" + "\n" + \
    #                "Connection: close" + "\n" + \
    #                "Last-Modified: " + parse_timestamp(timstamp) + "\n" + \
    #                "Expires: Mon, 20 Dec 1998 01:00:00 GMT" + "\n" + \
    #                "Content-Length: " + str(len((html).decode('windows-1252').encode('utf-8')) + 1) + "\n\n" + html.decode('windows-1252').encode('utf-8') + '\n\n'
    #
    #         record_str += str(len((next_str).decode('windows-1252').encode('utf-8')) + 1) + "\n\n" + next_str
    except Exception as e:
        print("Wrcer Prob - Docno:" + docno)
        with open('Prob.txt' ,'w') as f:
            f.write(html)
        record_str = ''

    return record_str


def create_warc_files_for_time_interval(
        destination_folder,
        time_interval,
        data_folder):

    num_of_records_in_interval = 0
    folder_files_hirarcy_dict = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            with open(os.path.join(data_folder, filename), 'r') as f:
                curr_json = f.read()
            curr_json = ast.literal_eval(curr_json)
            for snapshot in curr_json:
                if curr_json[snapshot]['RelevantInterval'] == time_interval:
                    docno = filename.replace('.json','')
                    inner_folder_name = docno.split('-')[1]
                    inner_file_name = docno.split('-')[2]
                    if inner_folder_name not in folder_files_hirarcy_dict:
                        folder_files_hirarcy_dict[inner_folder_name] = {}
                    if inner_file_name not in folder_files_hirarcy_dict[inner_folder_name]:
                        folder_files_hirarcy_dict[inner_folder_name][inner_file_name] = []

                    if len(curr_json[snapshot]['Url'].split('*/')) > 2:
                        raise Exception("create_warc_files_for_time_interval: Url has */ inside " +  str(curr_json[snapshot]['Url']))
                    else:
                        curr_url = curr_json[snapshot]['Url'].split('*/')[1]
                    curr_doc_representation = {'Docno'      : docno,
                                               'Url'        : curr_url,
                                               'TimeStamp'  : snapshot,
                                               'HTML'       : curr_json[snapshot]['html']}

                    folder_files_hirarcy_dict[inner_folder_name][inner_file_name].append(curr_doc_representation)
    print('Hirarchy Dict Finished...')
    for folder_name in folder_files_hirarcy_dict:
        for file_name in folder_files_hirarcy_dict[folder_name]:
            print("Creating Warc For " +folder_name + '-' + file_name)
            warc_str = ""
            warc_info_id = str(uuid.uuid1())
            warc_date = "2009-03-65T08:43:19-0800"
            warc_str += create_warc_head(
                    warc_date=warc_date,
                    warc_info_id=warc_info_id)
            for doc in folder_files_hirarcy_dict[folder_name][file_name]:
                curr_str = create_warc_record(
                    docno=doc['Docno'],
                    url=doc['Url'],
                    timstamp=doc['TimeStamp'],
                    html=doc['HTML'],
                    warc_date=warc_date,
                    warc_info_id=warc_info_id)
                if curr_str != '':
                    warc_str += curr_str
                    num_of_records_in_interval += 1
                else:
                    print('Docno: '  + doc['Docno'] + " Problematic")
            if not os.path.exists(os.path.join(destination_folder, time_interval, folder_name)):
                os.mkdir(os.path.join(destination_folder, time_interval, folder_name))
            with open(os.path.join(destination_folder, time_interval, folder_name, file_name + '.warc'), 'w') as f:
                f.write(warc_str)

    return num_of_records_in_interval

# if __name__ == '__main__':
#     work_year = '2008'
#     interval_list = build_interval_list(
#         work_year=work_year,
#         frequency='2W')
#
#     destination_folder = "/mnt/bi-strg3/v/zivvasilisky/data/2008/"
#     data_folder ="/lv_local/home/zivvasilisky/ziv/data/retrived_htmls/2008/"
#
#     summary_df = pd.DataFrame(columns = ['Interval', 'NumOfDocs'])
#     next_index = 0
#     for interval in interval_list:
#         print ("Curr interval: " + str(interval))
#         if not os.path.exists(os.path.join(destination_folder, interval)):
#             os.mkdir(os.path.join(destination_folder, interval))
#
#         num_records = create_warc_files_for_time_interval(
#             destination_folder=destination_folder,
#             time_interval=interval,
#             data_folder=data_folder)
#
#         summary_df.loc[next_index] = [interval, num_records]
#         next_index += 1
#
#     summary_df.to_csv(os.path.join(data_folder, 'Summry_warcer.tsv'), sep = '\t', index = False)


# test ####
# html = """<head> <meta http-equiv="Content-Language" content="en-gb"> <meta http-equiv="Content-Type" content="text/html; charset=windows-1252"> <title>00000-NRT-RealEstate's Homepage Startup</title> <base href="http://www.homepagestartup.com"> <meta name="keywords" content="startup homepage"> <meta name="description" content="00000-NRT-RealEstate's Homepage Startup. Your ideal start-up homepage."> <style type="text/css"> .rb { background-color:#FBFDFF; color:#A3A3A3;width:171px; height:128px; background-image: url('/img/rb.gif'); background-repeat: no-repeat;} .rbh { cursor:pointer; color:#750000; width:171px; height:128px; background-image: url('/img/rb.gif'); background-repeat: no-repeat; } body,font,a { font-family:arial; } form { margin-bottom:0px; } input.button { font: 14px arial,helvetica,sans-serif; 	font-weight: bold; padding: 3px 7px; 	_padding: 3px;  } input.cancel { font: 14px arial,helvetica,sans-serif; font-weight: bold; 	padding: 3px 7px; _padding: 3px; color:#666; } </style> <script language="javascript" src="/user/a.js"></script> </head> <body topmargin="0" leftmargin="0"> <div id="mainws"> <div align="center"> <div id="tp_div" align="right" style="position:absolute;width:100%;top:0px;left:0px;"> <table cellpadding="2" border="0" style="border-collapse: collapse"><tr><td> <a href="/" style="font-size:10pt;color:#880000;">go home</a></td></tr></table></div> <table border="0" style="border-collapse: collapse" cellspacing="2" cellpadding="2"> <tr> <td align="center" height="40"> <table border="0" style="border-collapse: collapse;z-index:10;" cellspacing="2" cellpadding="2"> <tr> <td height="25"> <a style="text-decoration:none;" href="http://00000-NRT-RealEstate.homepagestartup.com"><font style="letter-spacing: -0.5px;font-family: arial; font-weight:bold;" color="#750000" size="4">00000-NRT-RealEstate's</font></a> </td> <td><a href="http://00000-NRT-RealEstate.homepagestartup.com"><img style="z-index:100;position:relative;" border="0" src="/img/logo.gif"></a></td> </tr> </table> </td> </tr> <tr> <td align="center"> <form name="pf" action="/" style="margin-bottom:0px;"> <table border="1" style="border-collapse: collapse" id="tablesearch" cellspacing="0" bordercolor="#808080" cellpadding="0" bgcolor="#FFFFFF"> <tr> <td> <table border="1" style="border-collapse: collapse; background-image:url('img/bgshade.gif'); background-repeat:repeat-x" width="100%" cellpadding="0" bordercolor="#FFFFFF" cellspacing="0"> <tr> <td> <table border="0" style="border-collapse: collapse" cellpadding="0"> <tr> <td align="center" width="25"> <table cellpadding="0" style="border-collapse: collapse"><tr><td><a href="#" title="Click to change Search Engine" onclick="chs_shw();return false;"><div style="position:absolute;width:16px;height:16px;padding-top:12px;text-align:right;"><img src="img/ar.gif" border="0"></div></a><a href="#" title="Click to change Search Engine" onclick="chs_shw();return false;"><img border="0" src="/img/a.gif" name="simg" height="16" width="16"></a></td></tr></table></td> <td> <input type="text" value="&lt; search here &gt;" onblur="this.style.color='#5C5C5C';if(!this.value){this.value='< search here >';}" onfocus="this.style.color='#000000';if(this.value=='< search here >'){this.value='';}this.select();" name="q" size="51" style="border-width: 0px;padding-top:3px; font-size:10pt; font-weight:bold; height:22;color:#5C5C5C;background-color:transparent;"></td> <td width="63" align="center"> <input type="submit" style="width:60px;font:8pt verdana,arial;font-weight: bold;" value="Search"></td> </tr> </table> </td> </tr> </table> </td> </tr> </table> <input type="hidden" name="s" value="a"></form> </td> </tr> <tr> <td height="10"></td> </tr> <tr> <td align="center"><div id="websitedisplay"><table cellpadding="3" cellspacing="3"><tr><td><div style="width:171px; height:128px;" id="wsm1"><div id="ws1"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="NRT LLC" href="http://www.nrtinc.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.nrtinc.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">NRT LLC</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm2"><div id="ws2"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="Burgdorf.com" href="http://www.burgdorff.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.burgdorff.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">Burgdorf.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm3"><div id="ws3"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 3</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm4"><div id="ws4"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="Coldwell Banker Burnet" href="http://www.cbburnet.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbburnet.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">Coldwell Banker Burnet</font></td> </tr> </table> </td> </tr> </table></div></div></td></tr><tr><td><div style="width:171px; height:128px;" id="wsm5"><div id="ws5"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBDFW.com" href="http://www.cbdfw.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbdfw.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBDFW.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm6"><div id="ws6"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBMove.com" href="http://www.cbmove.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbmove.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBMove.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm7"><div id="ws7"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 7</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm8"><div id="ws8"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBSuccess.com" href="http://www.cbsuccess.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbsuccess.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBSuccess.com</font></td> </tr> </table> </td> </tr> </table></div></div></td></tr><tr><td><div style="width:171px; height:128px;" id="wsm9"><div id="ws9"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBWS.com" href="http://www.cbws.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbws.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBWS.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm10"><div id="ws10"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 10</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm11"><div id="ws11"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 11</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm12"><div id="ws12"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 12</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td></tr><tr></tr></table></div></td> </tr> <tr> <td align="center"> <table border="0" style="border-collapse: collapse" width="98%" id="table1"> <tr> <td> <b><font color="#484848" size="2">Welcome to 00000-NRT-RealEstate's Homepage</font></b></td> <td align="right"> <span style="font-size: 8pt"> <a style="color:#0000FF; font-family:arial" href="what_is_homepagestartup.html"> What is this?</a> | <a href="/" style="color:#0000FF; font-family:arial">Create Your Own Homepage</a></span></td> </tr> </table> </td> </tr> </table> </div> <div id="sbox" style="display:none;position:absolute;top:0px;left:0px;"> <table border="1" style="border-collapse: collapse" bordercolor="#A5ACB8" cellpadding="0" cellspacing="0"> <tr> <td bgcolor="#48505B" height="20" width="130"><font color="#FFFFFF"><span style="font-size: 8pt; font-weight: 700">&nbsp;Change My Search To:</span></font></td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('a');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/a.gif"></td> <td><font style="font-size: 8pt">Google Search</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('b');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/b.gif"></td> <td><font style="font-size: 8pt">MSN Search</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('c');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/c.gif"></td> <td><font style="font-size: 8pt">Yahoo! Search</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('d');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/d.gif"></td> <td><span style="font-size: 8pt">Ask.com Search</span></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('e');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/e.gif"></td> <td><span style="font-size: 8pt">Wikipedia English</span></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('f');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/c.gif"></td> <td><font style="font-size: 8pt">Yahoo! Answers</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('g');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/g.gif"></td> <td><span style="font-size: 8pt">Answers.com</span></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('h');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/h.gif"></td> <td><span style="font-size: 8pt">YouTube Videos</span></td> </tr> </table> </td> </tr> </table></div> </div> </body> </html>"""
# html = """
# <html xmlns:v="urn:schemas-microsoft-com:vml" xmlns:o="urn:schemas-microsoft-com:office:office" xmlns="http://www.w3.org/TR/REC-html40">
#
# <head>
# <meta http-equiv="Content-Type" content="text/html; charset=windows-1252">
# <meta http-equiv="Content-Language" content="en-us">
# <title>FIG Home Page</title>
# <meta name="GENERATOR" content="Microsoft FrontPage 6.0">
# <meta name="ProgId" content="FrontPage.Editor.Document">
# <link rel="stylesheet" type="text/css" href="fig.css">
# <SCRIPT language=JavaScript src="fig.js"></SCRIPT>
# <base target="_top">
# <style>
# <!--
# span.MsoHyperlink
# 	{color:blue;
# 	text-decoration:underline;
# 	text-underline:single;}
# -->
# </style>
# <meta name="Microsoft Border" content="none">
# </head>
#
# <body>
#   <center>
# 	<table border="0" cellpadding="3" cellspacing="3" width="610" height="100" id="table12">
# 		<tr>
# 			<td width="90" valign="top" align="left" rowspan="2" height="90">
# 			<img border="0" src="images/figlogo.gif" width="90" height="98"></td>
# 			<td valign="top" align="right" height="37"><font size="2"><b>
# 			<a onmouseover="ChkIn('Btn1')" onmouseout="ChkOut('Btn1')" href="general/profile.htm">
# 			<img border="0" src="images/aboutfig.gif" name="Btn1" width="80" height="14"></a>
# 			<a href="general/search.htm" onmouseover="ChkIn('Btn2')" onmouseout="ChkOut('Btn2')">
# 			<img border="0" src="images/search.gif" name="Btn2" width="80" height="14"></a>
# 			<a href="mailto:FIG@fig.net" onmouseover="ChkIn('Btn3')" onmouseout="ChkOut('Btn3')">
# 			<img border="0" src="images/feedback.gif" name="Btn3" width="80" height="14"></a>
# 			<script language="JavaScript"><!--
# function BrowserOK()
# {
# 	if (((navigator.appName == "Netscape") &&
# 	  (parseInt(navigator.appVersion) >= 3 )) ||
# 	  ((navigator.appName == "Microsoft Internet Explorer") &&
# 	  (parseInt(navigator.appVersion) >= 4 )))
# 		return true;
# 	else
# 		return false;
# }
#
# function ChkIn(str)
# {
# 	if (BrowserOK())
# 		ChkImageIn(str);
# }
# function ChkOut(str)
# {
# 	if (BrowserOK())
# 		ChkImageOut(str);
# }
# if (BrowserOK())
# 	{
# 	var str, strFile;
# 	str = document['Btn1'].src;
# 	strFile = str.substr(0, str.length - 4) + "_a.gif";
# 	Btn1n=BtnPreload(str);
# 	Btn1h=BtnPreload(strFile);
# 	str = document['Btn2'].src;
# 	strFile = str.substr(0, str.length - 4) + "_a.gif";
# 	Btn2n=BtnPreload(str);
# 	Btn2h=BtnPreload(strFile);
# 	str = document['Btn3'].src;
# 	strFile = str.substr(0, str.length - 4) + "_a.gif";
# 	Btn3n=BtnPreload(str);
# 	Btn3h=BtnPreload(strFile);
# 	}
# // --></script>
# 			</b></font></td>
# 			<td valign="top" align="right" rowspan="2">
# 			 ;</td>
# 		</tr>
# 		<tr>
# 			<td valign="top"><font size="2"><b>
# 			<img border="0" src="images/logotext.gif" width="347" height="42"><img border="0" src="images/coldummy.gif" width="67" height="2">
# 			</b></font></td>
# 		</tr>
# 	</table>
# <table border="0" cellpadding="3" cellspacing="3" style="border-collapse: collapse" bordercolor="#111111" width="687">
#   <tr>
#     <td>
#       <p class="fignavbar"><font size="2">
#       <img border="0" src="images/divider.gif" width="100%" height="14"></font><br>
#       <span style="text-transform: uppercase">
#       <!--webbot bot="Navigation" S-Type="children" S-Orientation="horizontal"
#       S-Rendering="text" B-Include-Home="FALSE" B-Include-Up="FALSE" U-Page
#       S-Target startspan --><nobr>[&nbsp;<a href="news/newsindex.htm">NEWS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="events/events.htm">EVENTS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="pub/index.htm">PUBLICATIONS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="annual_review/anrew-index.htm">ANNUAL&nbsp;REVIEW</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="admin/adminindex.htm">ADMINISTRATION</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="members/membindex.htm">MEMBERS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="corporatemembers/corporatemembers.htm">CORPORATE&nbsp;MEMBERS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="council/council_index.htm">COUNCIL</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="comm/comindex.htm">COMMISSIONS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="tf/tfindex.htm">TASK&nbsp;FORCES</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="perm/permanent.htm">PERMANENT&nbsp;INSTITUTIONS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="hsm/index.htm">HISTORY&nbsp;OF&nbsp;SURVEYING&nbsp;AND&nbsp;MEASUREMENT</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="figfoundation/index.htm">FIG&nbsp;FOUNDATION</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="standards_network/index.htm">STANDARDS&nbsp;NETWORK</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="underrep_groups/index.htm">UNDER-REPRESENTED&nbsp;GROUPS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="personalia/index.htm">PERSONALIA&nbsp;AND&nbsp;VISITS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="jobs/jobindex.htm">JOB&nbsp;SITE</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="admin/office.htm">FIG&nbsp;OFFICE</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="links/siteindex.htm">LINKS</a>&nbsp;]</nobr><!--webbot bot="Navigation" endspan i-checksum="37160" --> <font face="Arial">[ <a href="sedb/">FIG SURVEYING EDUCATION
#       DATABASE</a> ] [ <a href="discussion/discussion.asp">DISCUSSION GROUPS</a>
#       ]</font> <font face="Arial">[ <a href="http://www.habitatforum.org">
#       HABITAT PROFESSIONAL FORUM</a> ] [ <a href="http://www.fig.net/jbgis">
#       Joint BOARD OF GEOSPATIAL INFORMATION SOCIETIES</a> ] [ <a href="srl">SURVEYORS REFERENCE
#       LIBRARY </a>] [ <a href="council/president_enemark.htm">MEET THE PRESIDENT
#       </a>]</font></span></p>
#     </td>
#   </tr>
#   <tr>
#     <td bgcolor="#CC0000" height="22" style="border-bottom-style: none; border-bottom-width: medium">
#       <p class="figmainbar"><b><font size="5">Welcome to FIG Home Page</font></b></p>
#     </td>
#   </tr>
#   <tr>
#     <td style="border-style: none; border-width: medium" width="98%">
#       <h5 align="right" class="top"  ><a href="general/leaflet-french.htm">
#       <img border="0" src="general/france_10.jpg" width="15" height="10"></a>
#       <a href="general/leaflet-spanish.htm">
#       <img border="0" src="general/spain_10.jpg" width="15" height="10"></a>
#       <a href="general/leaflet_german_2003.htm">
#       <img border="0" src="general/germany_10.jpg" width="15" height="10"></a>
#       <a href="general/leaflet-russian.htm">
#       <img border="0" src="general/russia_10.jpg" width="15" height="10"></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; </h5>
#       <table border="0" width="90%" cellspacing="3" cellpadding="3" id="table25">
# 		<tr>
# 			<td width="31%" style="border-width: 1px" valign="top">
# 			<img border="0" src="images/christmas_2008.jpg" width="200" height="140"></td>
# 			<td width="65%">
# 			<i>International Federation of Surveyors brings you Season's
# 			Greetings and wishes you a prosperous New Year 2009. Thank you for
# 			your co-operation in 2008. We look forward to meeting you at the FIG
# 			Working Week in Eilat in May, at the Regional Conference in Hanoi in
# 			October or at one of our commission events.</i><p><i>FIG Council and
# 			FIG Office</i></p>
# 			<p><i>PS. The FIG Office will be closed <b>23 December - 4 January.</b></i><br>
# &nbsp;</td>
# 		</tr>
# 		<tr>
# 			<td colspan="2">
# 			<table border="0" width="623" cellspacing="3" cellpadding="3" id="table26" height="267">
# 				<tr>
# 					<td width="387" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" bgcolor="#FFCCCC">
# 					<p align="center" class="figtext"><b>
# 					<font style="font-size: 9pt">Major FIG Conferences</font></b></td>
# 					<td width="211" colspan="2" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" bgcolor="#FFCCCC">
# 					<p align="center" class="figtext"><b>
# 					<font style="font-size: 9pt">FIG Platinum Corporate Members</font></b></td>
# 				</tr>
# 				<tr>
# 					<td width="376" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" rowspan="3" valign="top">
# 					<div align="center">
# 						<table border="0" width="102%" id="table27" cellpadding="2">
# 							<tr>
# 								<td>
# 								<p align="center">
# 								<a href="http://www.fig.net/fig2009">
# 								<img border="0" src="events/2009/fig2009_60.jpg" width="200" height="60" style="border: 1px solid #C0C0C0" alt="FIG Working Week 2009 - Eilat, Israel, 3-8 May 2009"></a></td>
# 							</tr>
# 							<tr>
# 								<td>
# 								<p align="center">
# 								<a href="http://www.fig.net/wb2009">
# 								<img border="0" src="events/2009/wb_60.jpg" width="355" height="60" style="border: 1px solid #C0C0C0" alt="FIG / World Bank International Conference - Washington DC, USA, 9-10 March 2010"></a></td>
# 							</tr>
# 							<tr>
# 								<td>
# 								<p align="center">
# 								<a href="http://www.fig.net/vietnam">
# 								<img border="0" src="events/2009/vietnam_60.jpg" width="332" height="60" style="border: 1px solid #C0C0C0" alt="7th FIG Regional Conference - Hanoi, Vietnam, 19-22 October 2009"></a></td>
# 							</tr>
# 							<tr>
# 								<td>
# 								<p align="center">
# 								<a href="http://www.fig2010.com">
# 								<img border="0" src="events/2009/sydney_140x70.jpg" width="140" height="68" style="border: 1px solid #C0C0C0" alt="XXIV FIG International Congress - Sydney, Australia, 11-16 April 2010"></a></td>
# 							</tr>
# 						</table>
# 					</div>
# 					</td>
# 					<td width="100" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" height="82">
# 					<a href="http://www.bentley.com/">
# 					<img border="0" src="corporatemembers/logos/bentley_100.jpg" width="100" height="32"></a></td>
# 					<td width="100" align="center" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" height="82">&nbsp;&nbsp;
# 					<a href="http://www.esri.com/">
# 					<img border="0" src="corporatemembers/logos/esri_100.jpg" width="49" height="60"></a></td>
# 				</tr>
# 				<tr>
# 					<td width="100" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" height="79">
# 					<a href="http://www.intergraph.com/">
# 					<img border="0" src="corporatemembers/logos/intergraph_100.jpg" width="100" height="28"></a></td>
# 					<td width="100" align="center" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0" height="79">&nbsp;
# 					<a href="http://www.leica-geosystems.com/corporate/en/lgs_405.htm">
# 					<img border="0" src="corporatemembers/logos/leica_80.jpg" width="80" height="51"></a></td>
# 				</tr>
# 				<tr>
# 					<td width="100" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0">
# 					<a href="http://www.topcon.co.jp/eng/">
# 					<img border="0" src="corporatemembers/logos/topcon_100.jpg" width="100" height="31" vspace="12"></a></td>
# 					<td width="100" align="center" style="border-style: solid; border-width: 1px" bordercolor="#C0C0C0">
# 					<a href="http://www.trimble.com">
# 					<img border="0" src="corporatemembers/logos/trimble_100.jpg" width="100" height="25" vspace="12"></a></td>
# 				</tr>
# 			</table></td>
# 		</tr>
# 		</table>
# 		<ul>
# 			<li>
# 			<h5  ><a href="events/events2009.htm">FIG organised or co-sponsored events
#     this and coming months</a></h5></li>
# 			<li>
# 			<h5  >
# 			<a href="pub/monthly_articles/december_2008/december_2008_zimmermann.html">Article of the Month -
# 			December 2008</a></h5>
# 			<p  >In the FIG <i><a href="pub/monthly_articles/index.htm">Article of the Month</a></i>
#       		series high-level papers focusing on interesting topics to all
# 			surveyors are published. The paper can be picked up from FIG
# 			conference or another event. The Article of the Month in December
# 			2008 is &quot;<a href="pub/monthly_articles/december_2008/december_2008_zimmermann.html">Effective
# 			and Transparent Management of Public Land</a>&quot;. It is written by Mr.
# 			<b>Willi Zimmermann</b> from Germany. His paper is an updated
# 			version of the paper that has been presented at the FIG/FAO/CNG
# 			International Seminar on State and Public Land Management in Verona,
# 			Italy, 9-10 September 2008. </p></li>
# 			<li>
# 			<h5  ><a href="fig2009">FIG Working Week 2009 - Surveyors Key Role
# 			in Accelerated Development,</a><br>
# 			<a href="fig2009">Eilat, Israel
# 		- 3-8 May 2009</a></h5></li>
# 		</ul>
# 		<blockquote>
# 			<p  >The organisers of FIG Working Week 2009 have received almost
# 			300 proposals for presentations. Deadline for abstracts&nbsp;for non
# 			peer review papers has passed
# 			but still some late coming proposals will be considered. The authors
# 			will be informed about their papers in early January and the
# 			programme will be published mid January 2009.</p>
# 			<table border="0" width="591" cellspacing="3" cellpadding="3" id="table22">
# 				<tr>
# 					<td width="150" valign="top">
# 					<a href="events/2009/2nd_invitation.pdf">
# 					<img border="0" src="fig2009/images/israel_150.jpg" width="150" height="105"></a></td>
# 					<td>
# 					<ul>
# 						<li>Submit your non peer review abstract online at:
# 						<a href="http://www.fig.net/abstractdb/submit.asp?id=10">
# 						www.fig.net/abstractdb/submit.asp?id=10</a>. Deadline
# 						for abstracts&nbsp; has passed but
# 						some late coming proposals will still be considered.</li>
# 						<li>Call for papers:
# 						<a href="http://www.fig.net/fig2009/call.htm">
# 						www.fig.net/fig2009/call.htm</a> </li>
# 						<li>Conference web site:
# 						<a href="http://www.fig.net/fig2009">www.fig.net/fig2009</a>
# 						- register now at:
# 						<a href="https://www.ortra.com/fig/default-register_r.htm">
# 						https://www.ortra.com/fig/default-register_r.htm</a></li>
# 					</ul>
# 					</td>
# 				</tr>
# 			</table>
# 		</blockquote>
# 		<ul>
# 			<li>
# 			<h5  ><a href="pub/enews/index.htm">FIG e-Newsletter</a></h5>
# 			<p  >You can download
#         the latest FIG e-Newsletter - <a href="pub/enews/december_2008.pdf">
# 			December 2008</a><a href="pub/enews/march_2008.pdf">
# 			</a>from here as a .pdf-file.
#         If you do not already get your copy of the e-Newsletter you can order it
#         now from
#         	<a href="http://www.fig.net/pub/subscriptions/getnewsletter.htm">http://www.fig.net/pub/subscriptions/getnewsletter.htm</a>.</p>
# 			</li>
# 			<li>
# 			<h5  ><a href="news/news_2008/pasadena_december_2008.htm">FIG Vice
# 			President Matt Higgins attends the Third Meeting of the
# 			International Committee on Global Navigation Satellite Systems (ICG)
# 			- Pasadena, California, USA, 8-12 December 2008</a></h5></li>
# 		</ul>
# 		<blockquote>
# 			<p  >The International Committee on Global Navigation Satellite
# 			Systems (ICG), met in Pasadena, California, USA, from 8 to 12
# 			December 2008. Vice President <b>Matt Higgins</b> attended
# 			representing the International Federation of Surveyors (FIG). <i>
# 			<a href="news/news_2008/pasadena_december_2008.htm">Read more...</a></i></p>
# 		</blockquote>
# 		<ul>
# 			<li>
# 			<h5  ><a name="John">John</a> Neel appointed as General Manager of FIG </h5></li>
# 		</ul>
# 		<blockquote>
# 			<table border="0" width="592" cellspacing="3" cellpadding="3" id="table30">
# 				<tr>
# 					<td valign="top" width="0">
# 					<img border="0" src="office/neel_120_x180.jpg" width="120" height="181"></td>
# 					<td width="409">
# 					<p>Mr. <b>John Neel </b>from Denmark has been appointed
# 						to the General Manager of FIG. John was selected among a
# 						range of qualified candidates. As General Manager he
# 						will have complete responsibility for the management and
# 						administration of the FIG Office. The current FIG
# 						Director, <b>Markku Villikka</b>, will then have a more
# 						advisory role to the President and Council including
# 						development of the FIG events. John Neel will start in
# 						the FIG Office by 1 January 2009 and can be contacted by
# 						email: <a href="mailto:john.neel@fig.net">john.neel@fig.net</a>.
# 						</p>
# 					<p>John Neel, 57, has a background in international
# 						banking and consulting, with working experience from
# 						several overseas assignments. His latest international
# 						posting was as head of an EU investment promotion
# 						programme in Kosovo. Prior to that he was head of
# 						Danida’s B-2-B (Business to Business) Programme, first
# 						in Egypt, and then in Kenya. Through most of his career
# 						he has been involved with organisation development,
# 						business expansion, and planning and implementation of
# 						international projects.</p></td>
# 				</tr>
# 			</table>
# 		</blockquote>
# 		<ul>
# 			<li>
# 			<h5><a href="news/news_2008/rome_nov_2008.htm">FIG
# 						President Stig Enemark attends Expert Group Meeting on
# 						Voluntary Guidelines on Responsible Governance of Tenure
# 						of Land and Natural Resources - FAO Headquarters, Rome,
# 						Italy, 24-25 November 2008</a></h5>
# 			</li>
# 		</ul>
# 		<blockquote>
# 			<table border="0" width="595" cellspacing="3" cellpadding="3" id="table31">
# 				<tr>
# 					<td width="136" valign="top">
# 					<a href="news/news_2008/rome_november_2008/fao_flags_1000.jpg">
# 					<img border="0" src="news/news_2008/rome_november_2008/fao_flags_nov_2008_120.jpg" width="120" height="128" alt="Click picture for bigger format."></a></td>
# 					<td valign="top">
# 					<p>FIG President <b>Stig Enemark</b> attended the Expert Group Meeting on
# 	Voluntary Guidelines on Responsible Governance of Tenure of Land and other
# 	Natural Resources at FAO Headquarters, Rome, Italy, 24–25 November 2008.&nbsp; FAO, and partners such as the World Bank, UN-HABITAT, IFAD
# 					(International Federation of Agricultural Development), and
# 					FIG have been working since 2005 on governance of land
# 					administration to raise awareness and produce guidelines.<i>
# 					<a href="news/news_2008/rome_nov_2008.htm">Read more...</a></i></p>
# 					</td>
# 				</tr>
# 			</table>
# 		</blockquote>
# 		<ul>
# 			<li>
# 			<h5>Nomination for Chair Elect of Commission 8 and Commission 10 </h5></li>
# 		</ul>
# 		<blockquote>
# 			<p>The General Assembly elected Chairs Elect to 8 FIG Commissions in
# 			Stockholm in June 2008. Election or appointment of Chair Elect to
# 			Commission 8 (Spatial Planning and Development) and Commission 10
# 			(Construction Economics and Management) was postponed to the General
# 			Assembly in May 2009. Member associations have been invited to make
# 			nominations for Chair Elect to these two commissions. Closing date
# 			for nomination is <b>3 January 2009</b>. The term of office for the
# 			Chair Elect will be from the appointment to the end of 2010. Under
# 			normal circumstances the Chair Elect will be appointed to the Chair
# 			of his/her Commission at the General Assembly in Sydney in April
# 			2010 and the term of office will be 1.1.2011-31.12.2014. Nominations
# 			will not be considered after the closing date. </p>
# 			<blockquote>
# 				<ul>
# 					<li>
# 					<a href="news/news_2008/nominations_2009/template_chairelect_2009.doc">
# 					Template for nomination of a Chair Elect as a .doc-file</a></li>
# 					<li>
# 					<a href="news/news_2008/nominations_2009/template_chairelect_2009.pdf">
# 					Template for nomination of a Chair Elect as a .pdf-file</a></li>
# 					<li>
# 					<a href="news/news_2008/nominations_2009/rules_chairelect_2009.pdf">
# 					FIG Internal Rules related to election of Chair Elects</a></li>
# 				</ul>
# 			</blockquote>
# 		</blockquote>
# 		<ul>
# 			<li>
# 			<h5  >
# 			<a href="events/2009/history_2009.htm">Exhibition on the
# 			Evolution from Local Measures to the Meter - Braine-l’ Alleud,
# 			Belgium, 9 January – 7 February 2009 </a></h5></li>
# 			<li>
# 			<h5  >
# 			<a href="http://www.i3mainz.fh-mainz.de/FIG-Workshop/">FIG
# 			Commission 3 Workshop on “Spatial Information for Sustainable
# 			Management of Urban Areas” -&nbsp; 2 - 4 February, Mainz, Germany</a></h5></li>
# 		</ul>
# 		<blockquote>
# 			<p  >Web site:
# 			<a href="http://www.i3mainz.fh-mainz.de/FIG-Workshop/">
# 			http://www.i3mainz.fh-mainz.de/FIG-Workshop/</a>&nbsp; and
# 			<a href="events/2009/com_3_february_2009.pdf">Call for Papers</a><br>
# 			Deadline for short abstract submission (full paper refereed channel)
# 			and for extended abstract submission (abstract refereed channel) has
# 			passed, but you can contact the organisers for last minute
# 			proposals. </p>
# 		</blockquote>
# 		<ul>
# 			<li>
# 			<h5  ><a href="jobs/jobindex.htm">Vacancies and project opportunities
# 			</a>(latest updated
# 			22 December 2008)</h5></li>
# 			<li>
# 			<h5  ><a href="personalia/index.htm">Personalia and visits</a> (last
# 			updated 30 October 2008)</h5></li>
# 			<li>
# 			<h5  ><a href="corporatemembers/corporate_benefits.htm">Benefits
# 			from FIG Corporate Members to FIG members and members of FIG member
# 			associations</a> - Get now free subscription of GIM International</h5></li>
# 			<li>
# 			<h5  >FIG Surveying Education Database (SEDB)&nbsp;</h5></li>
# 		</ul>
#
#   <blockquote>
#     <table border="0" style="border-collapse: collapse" bordercolor="#111111" cellpadding="3" cellspacing="3">
#       <tr>
#         <td valign="top"><img border="0" src="images/owl-small.jpg" width="150" height="184">
#           </td>
#         <td>The <a href="sedb/index.htm">Surveying Education Database</a> (SEDB)
#         on Professional Education is in full operation. Universities and
#         institutions are able with their user-ID and password to make changes
#         directly to the database.&nbsp;Any academic department offering graduate and post-graduate courses
#         in any surveying discipline can place a standard entry on the SEDB.
#           <p>The Surveying Education Database is a major benefit for
#       being an&nbsp; <a href="sedb/am_overview.htm">Academic Member of FIG</a>.
#       Academic Members of FIG can add a picture and additional information to
#       their standard entries.
#         </td>
#       </tr>
#     </table>
#   </blockquote>
#
#   <ul>
#     <li>
#       <h5><a href="news/news_2008.htm">Other news from the last months </a>(last
# 		updated 17 August 2008)and
# 		<a href="news/news_shortstories.htm">World around FIG - News</a> (last
# 		updated 4 July 2008)</h5>
#     </li>
#   </ul>
#   <ul>
#     <li><h5>Latest FIG Publications and Newsletters</h5>
#       <ul>
#         <li>
#         <a href="commission5/newsletters/december_%202008.pdf">Commission 5
# 		Newsletter - December 2008</a></li>
# 		<li>
#         <a href="commission1/newsletters/ys_newsletter_dec_2008.pdf">Young Surveyors Newsletter -
# 		December 2008</a></li>
# 		<li>
#         <a href="pub/figpub/pub44/figpub44.htm">FIG Publication no. 44 -&nbsp;
# 		Improving Slum Conditions through Innovative Financing. FIG/UN-HABITAT
# 		Seminar, Stockholm, Sweden, 16–17 June 2008. Summary Report.</a> FIG
# 		Report.</li>
# 			<li>
# 			<a href="commission7/verona_fao_2008/index.htm">Proceedings of the
# 			FIG/FAO/CNG Seminar on State and Public Land Management.</a> Verona,
# 			Italy, 9-10 September 2008</li>
# 		<li><a href="commission7/verona_am_2008/index.htm">Proceedings of FIG
# 		Commission 7 Annual Meeting and FIG/CNG Open Symposium on Environment
# 		and Land Administration &quot;Big Works for the Defense of the Territory&quot;.</a>
# 		Verona 11-15 September 2008.</li>
# 		<li>
#         <a href="pub/commissions/commission7/com_7_newsletter_oct_2008.pdf">
# 		Commission 7 Newsletter - October 2008</a></li>
# 		<li>
#         <a href="commission6/lisbon_2008/index.htm">Proceedings of the Joint
# 		Symposium “Measuring the Changes” of the FIG 13th Symposium on
# 		Deformation Measurement and Analysis and the IAG 4th Symposium on
# 		Geodesy for Geotechnical and Structural Engineering</a> - Lisbon,
# 		Portugal, 12-15 May 2008.</li>
# 		<li>
#         <a href="commission7/seoul_2007/index.htm">Proceedings of the FIG
# 		Commission 7 Annual Meeting 2007 and Symposium on “Good Practice in
# 		Cadastre and Land Registry”</a> - Seoul, Republic of Korea, 19 - 23 May
# 		2007.</li>
# 		<li>
#         <a href="pub/figpub/pub43_span/figpub43.htm">Publicaciףn de la FIG No 43
# 		- Declaraciףn de Costa Rica - Gestiףn de zonas costeras a favor de los
# 		pobres.</a> Declaraciףn de principios de la FIG, 2008.</li>
# 		<li>
#         <a href="commission2/enschede_2008/index.html">FIG International
# 		Workshop - Sharing Good Practices: E-learning in Surveying,
# 		Geo-Information Sciences and Land Administration</a>. ITC, Enschede, The
# 		Netherlands, 11-13 June 2008 organised by FIG Commission 2, ITC and Kadaster, The Netherlands. Proceedings of the workshop.</li>
# 		<li>
#         <a href="pub/figpub/pub40/figpub40.htm">FIG Publication no. 40 -
#           FIG Statutes, Internal Rules and Guidelines.</a>&nbsp; FIG
# 		Regulations.</li>
# 		<li>
#         <a href="commission6/newsletters/commission_6_newsletter_may_2008.pdf">Commission
# 		6 Newsletter</a>, May 2008</li>
# 		<li>
#         <a href="pub/figpub/pub_flyerfinal.pdf">FIG Publications - The
# 		development of policies, profession, and practice</a>. Flyer as a .pdf-file. </li>
# 		<li>
#         <a href="pub/figpub/pub43/figpub43.htm">FIG Publication no. 43 - Costa
# 		Rica Declaration on Pro-Poor Coastal Zone Management.</a> FIG Policy
# 		Statement. </li>
# 		<li>
#         <a href="general/profile.htm">The FIG Profile and the benefits of being
# 		a member 2007 - 2010</a>. FIG leaflet in
# 		<a href="general/profile/profile_250808.pdf">.pdf-format</a>. </li>
# 		<li>
#         <a href="pub/figpub/pub42/figpub42.htm">FIG Publication no. 42 -
#           Informal Settlements: The Road towards More Sustainable Places.</a>
# 		FIG Report.</li>
# 		<li>
#         <a href="pub/figpub/pub41/figpub41.htm">FIG Publication no. 41 -
# 		Capacity Assessment in Land Administration.</a> FIG Guide.
# 		</li>
# 		<li>
#         <a href="pub/figpub/pub39/figpub39.htm">FIG Publication no. 39 - FIG
# 		Work Plan 2007-2010.</a> FIG Regulations. </li>
# 		<li>
#         <a href="pub/underrep_news/200801/newsletter200801.htm">Under-represented Groups in FIG - Newsletter 1/2008</a></li>
# 		<li>
#         <a href="annual_review/anrew06_07/anrev06_07.htm">FIG Annual Review June
# 		2006 - December 2007</a></li>
#       </ul>
#     </li>
#   </ul>
#   <hr color="#CC0000">
#   <P><strong>&nbsp; </strong>
#   <!--webbot bot="HitCounter"
#   u-custom="images/counter.gif" i-digits="5" i-image="5"
#   PREVIEW="&lt;strong&gt;[Hit Counter]&lt;/strong&gt;" i-resetvalue="29860" startspan --><img src="_vti_bin/fpcount.exe/?Page=indexmain.htm|Custom=images/counter.gif|Digits=5" alt="Hit Counter"><!--webbot bot="HitCounter" endspan i-checksum="46263" -->
#   </P>
#   <table border="0" cellpadding="3" cellspacing="3" style="border-collapse: collapse" bordercolor="#111111" width="98%">
#     <tr>
#       <td width="305" valign="top"><FONT face=Arial><font color="#990000"><b>
#       <a href="admin/office.htm">FIG
#         Office</a></b></font><br>
#       </FONT>International Federation of Surveyors, FIG<br>
#       Kalvebod Brygge 31-33<br>
#       DK-1780 Copenhagen V<br>
#       DENMARK<br>
#       e-mail: <a href="mailto:FIG@fig.net">FIG@fig.net</a> <br>
#       Tel + 45 3886 1081<br>
#       Fax + 45 3886 0252</td>
#       <td valign="top"><FONT face=Arial><b>FIG Director<font color="#990000"><br>
#         </font></b>Mr. Markku Villikka<br>
#         e-mail <a href="mailto:markku.villikka@fig.net">markku.villikka@fig.net</a></FONT><br>
#       <FONT face=Arial>Tel. + 358 44 357 0911 (direct)<br>
#       <b>Personal
#       Assistant</b><br>
#         Ms.
#         Tine Svendstorp<br>
#         e-mail <span lang="fi"><a href="mailto:tine.svendstorp@fig.net">
#       tine.svendstorp@fig.net</a> <br>
#       Tel. + 45 3318 5584 (direct)</span></FONT></td>
#     </tr>
#   </table>
#     </td>
#   </tr>
#   <tr>
#     <td height="44" style="border-top-style: none; border-top-width: medium">
#       <p class="fignavbar"><font size="2">
#       <img border="0" src="images/divider.gif" width="100%" height="14"></font><font size="1"><b><font color="#990000"><br>
#       </font></b></font>
#       <!--webbot bot="Navigation" S-Type="children"
#       S-Orientation="horizontal" S-Rendering="text" B-Include-Home="FALSE"
#       B-Include-Up="FALSE" U-Page S-Target startspan --><nobr>[&nbsp;<a href="news/newsindex.htm">NEWS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="events/events.htm">EVENTS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="pub/index.htm">PUBLICATIONS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="annual_review/anrew-index.htm">ANNUAL&nbsp;REVIEW</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="admin/adminindex.htm">ADMINISTRATION</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="members/membindex.htm">MEMBERS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="corporatemembers/corporatemembers.htm">CORPORATE&nbsp;MEMBERS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="council/council_index.htm">COUNCIL</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="comm/comindex.htm">COMMISSIONS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="tf/tfindex.htm">TASK&nbsp;FORCES</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="perm/permanent.htm">PERMANENT&nbsp;INSTITUTIONS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="hsm/index.htm">HISTORY&nbsp;OF&nbsp;SURVEYING&nbsp;AND&nbsp;MEASUREMENT</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="figfoundation/index.htm">FIG&nbsp;FOUNDATION</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="standards_network/index.htm">STANDARDS&nbsp;NETWORK</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="underrep_groups/index.htm">UNDER-REPRESENTED&nbsp;GROUPS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="personalia/index.htm">PERSONALIA&nbsp;AND&nbsp;VISITS</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="jobs/jobindex.htm">JOB&nbsp;SITE</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="admin/office.htm">FIG&nbsp;OFFICE</a>&nbsp;]</nobr> <nobr>[&nbsp;<a href="links/siteindex.htm">LINKS</a>&nbsp;]</nobr><!--webbot bot="Navigation" endspan i-checksum="37160" --><font face="Arial"><span style="text-transform: uppercase">
#       </span></font>
#       <span style="text-transform: uppercase">
#       <font face="Arial">[ <a href="sedb/">FIG SURVEYING EDUCATION DATABASE</a>
#       ] [ <a href="discussion/discussion.asp">DISCUSSION GROUPS</a> ] [ <a href="http://www.habitatforum.org">HABITAT
#       PROFESSIONAL FORUM</a> ] [ <a href="http://www.fig.net/jbgis">
#       Joint BOARD OF GEOSPATIAL INFORMATION SOCIETIES</a> ] [ <a href="srl">SURVEYORS REFERENCE LIBRARY </a>] [
#       <a href="council/president_enemark.htm">MEET THE PRESIDENT</a> ]</font></span><b><br>
#       </b>
#       <font color="#000000">This page is maintained by <a href="mailto:FIG@fig.net">the
#       FIG Office</a>. Last revised</font> on
#       <!--webbot
#       bot="Timestamp" S-Type="EDITED" S-Format="%Y-%m-%d" startspan -->2008-12-30<!--webbot
#       bot="Timestamp" endspan i-checksum="12181" -->.</p>
#     </td>
#   </tr>
# </table>
#   </center>
#
#   </body></html>"""
# print (create_warc_record(
#     docno='clueweb09-en0000-00-00000',
#     timstamp="2009-01-13 18:05:10",
#     url = 'http://00000-nrt-realestate.homepagestartup.com/',
#     html=html,
#     warc_info_id ='993d3969-9643-4934-b1c6-68d4dbe55b83',
#     warc_date='2009-03-65T08:43:19-0800'))

# print (create_warc_head(warc_date="2009-03-65T08:43:19-0800",
#                         warc_info_id='993d3969-9643-4934-b1c6-68d4dbe55b83'))


