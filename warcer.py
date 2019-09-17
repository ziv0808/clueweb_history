import os
import ast
import uuid
import pandas as pd


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
    except Exception as e:
        # print (html)
        try:
            next_str = "HTTP/1.1 200 OK" + "\n" + \
                   "Content-Type: text/html" + "\n" + \
                   "Date: " + parse_timestamp(timstamp) + "\n" + \
                   "Pragma: no-cache" + "\n" + \
                   "Cache-Control: no-cache, must-revalidate" + "\n" + \
                   "X-Powered-By: PHP/4.4.8" + "\n" + \
                   "Server: WebServerX" + "\n" + \
                   "Connection: close" + "\n" + \
                   "Last-Modified: " + parse_timestamp(timstamp) + "\n" + \
                   "Expires: Mon, 20 Dec 1998 01:00:00 GMT" + "\n" + \
                   "Content-Length: " + str(len((html).decode('windows-1252').encode('utf-8')) + 1) + "\n\n" + html + '\n\n'

            record_str += str(len((next_str).decode('windows-1252').encode('utf-8')) + 1) + "\n\n" + next_str
        except Exception as e:
            with open('Prob.txt' ,'w') as f:
                f.write(html)

            raise e
    return record_str


def create_warc_files_for_time_interval(
        destination_folder,
        time_interval,
        data_folder):

    num_of_records_in_interval = 0
    folder_files_hirarcy_dict = {}
    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            with open(os.path.join(data_folder, filename), 'rb') as f:
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
                warc_str += create_warc_record(
                    docno=doc['Docno'],
                    url=doc['Url'],
                    timstamp=doc['TimeStamp'],
                    html=doc['HTML'],
                    warc_date=warc_date,
                    warc_info_id=warc_info_id)
                num_of_records_in_interval += 1
            if not os.path.exists(os.path.join(destination_folder, time_interval, folder_name)):
                os.mkdir(os.path.join(destination_folder, time_interval, folder_name))
            with open(os.path.join(destination_folder, time_interval, folder_name, file_name + '.warc'), 'w') as f:
                f.write(warc_str)

    return num_of_records_in_interval

if __name__ == '__main__':
    work_year = '2008'
    interval_list = build_interval_list(
        work_year=work_year,
        frequency='2W')

    destination_folder = "/mnt/bi-strg3/v/zivvasilisky/data/2008/"
    data_folder ="/lv_local/home/zivvasilisky/ziv/data/retrived_htmls/2008/"

    summary_df = pd.DataFrame(columns = ['Interval', 'NumOfDocs'])
    next_index = 0
    for interval in interval_list:
        print ("Curr interval: " + str(interval))
        if not os.path.exists(os.path.join(destination_folder, interval)):
            os.mkdir(os.path.join(destination_folder, interval))

        num_records = create_warc_files_for_time_interval(
            destination_folder=destination_folder,
            time_interval=interval,
            data_folder=data_folder)

        summary_df.loc[next_index] = [interval, num_records]
        next_index += 1

    summary_df.to_csv(os.path.join(data_folder, 'Summry_warcer.tsv'), sep = '\t', index = False)


## test ####
# html = """<head> <meta http-equiv="Content-Language" content="en-gb"> <meta http-equiv="Content-Type" content="text/html; charset=windows-1252"> <title>00000-NRT-RealEstate's Homepage Startup</title> <base href="http://www.homepagestartup.com"> <meta name="keywords" content="startup homepage"> <meta name="description" content="00000-NRT-RealEstate's Homepage Startup. Your ideal start-up homepage."> <style type="text/css"> .rb { background-color:#FBFDFF; color:#A3A3A3;width:171px; height:128px; background-image: url('/img/rb.gif'); background-repeat: no-repeat;} .rbh { cursor:pointer; color:#750000; width:171px; height:128px; background-image: url('/img/rb.gif'); background-repeat: no-repeat; } body,font,a { font-family:arial; } form { margin-bottom:0px; } input.button { font: 14px arial,helvetica,sans-serif; 	font-weight: bold; padding: 3px 7px; 	_padding: 3px;  } input.cancel { font: 14px arial,helvetica,sans-serif; font-weight: bold; 	padding: 3px 7px; _padding: 3px; color:#666; } </style> <script language="javascript" src="/user/a.js"></script> </head> <body topmargin="0" leftmargin="0"> <div id="mainws"> <div align="center"> <div id="tp_div" align="right" style="position:absolute;width:100%;top:0px;left:0px;"> <table cellpadding="2" border="0" style="border-collapse: collapse"><tr><td> <a href="/" style="font-size:10pt;color:#880000;">go home</a></td></tr></table></div> <table border="0" style="border-collapse: collapse" cellspacing="2" cellpadding="2"> <tr> <td align="center" height="40"> <table border="0" style="border-collapse: collapse;z-index:10;" cellspacing="2" cellpadding="2"> <tr> <td height="25"> <a style="text-decoration:none;" href="http://00000-NRT-RealEstate.homepagestartup.com"><font style="letter-spacing: -0.5px;font-family: arial; font-weight:bold;" color="#750000" size="4">00000-NRT-RealEstate's</font></a> </td> <td><a href="http://00000-NRT-RealEstate.homepagestartup.com"><img style="z-index:100;position:relative;" border="0" src="/img/logo.gif"></a></td> </tr> </table> </td> </tr> <tr> <td align="center"> <form name="pf" action="/" style="margin-bottom:0px;"> <table border="1" style="border-collapse: collapse" id="tablesearch" cellspacing="0" bordercolor="#808080" cellpadding="0" bgcolor="#FFFFFF"> <tr> <td> <table border="1" style="border-collapse: collapse; background-image:url('img/bgshade.gif'); background-repeat:repeat-x" width="100%" cellpadding="0" bordercolor="#FFFFFF" cellspacing="0"> <tr> <td> <table border="0" style="border-collapse: collapse" cellpadding="0"> <tr> <td align="center" width="25"> <table cellpadding="0" style="border-collapse: collapse"><tr><td><a href="#" title="Click to change Search Engine" onclick="chs_shw();return false;"><div style="position:absolute;width:16px;height:16px;padding-top:12px;text-align:right;"><img src="img/ar.gif" border="0"></div></a><a href="#" title="Click to change Search Engine" onclick="chs_shw();return false;"><img border="0" src="/img/a.gif" name="simg" height="16" width="16"></a></td></tr></table></td> <td> <input type="text" value="&lt; search here &gt;" onblur="this.style.color='#5C5C5C';if(!this.value){this.value='< search here >';}" onfocus="this.style.color='#000000';if(this.value=='< search here >'){this.value='';}this.select();" name="q" size="51" style="border-width: 0px;padding-top:3px; font-size:10pt; font-weight:bold; height:22;color:#5C5C5C;background-color:transparent;"></td> <td width="63" align="center"> <input type="submit" style="width:60px;font:8pt verdana,arial;font-weight: bold;" value="Search"></td> </tr> </table> </td> </tr> </table> </td> </tr> </table> <input type="hidden" name="s" value="a"></form> </td> </tr> <tr> <td height="10"></td> </tr> <tr> <td align="center"><div id="websitedisplay"><table cellpadding="3" cellspacing="3"><tr><td><div style="width:171px; height:128px;" id="wsm1"><div id="ws1"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="NRT LLC" href="http://www.nrtinc.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.nrtinc.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">NRT LLC</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm2"><div id="ws2"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="Burgdorf.com" href="http://www.burgdorff.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.burgdorff.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">Burgdorf.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm3"><div id="ws3"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 3</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm4"><div id="ws4"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="Coldwell Banker Burnet" href="http://www.cbburnet.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbburnet.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">Coldwell Banker Burnet</font></td> </tr> </table> </td> </tr> </table></div></div></td></tr><tr><td><div style="width:171px; height:128px;" id="wsm5"><div id="ws5"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBDFW.com" href="http://www.cbdfw.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbdfw.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBDFW.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm6"><div id="ws6"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBMove.com" href="http://www.cbmove.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbmove.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBMove.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm7"><div id="ws7"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 7</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm8"><div id="ws8"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBSuccess.com" href="http://www.cbsuccess.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbsuccess.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBSuccess.com</font></td> </tr> </table> </td> </tr> </table></div></div></td></tr><tr><td><div style="width:171px; height:128px;" id="wsm9"><div id="ws9"><table cellspacing="0" cellpadding="0"> <tr> <td class="rb" align="center" style="padding-top:7px;" valign="top"> <table border="0" style="border-collapse: collapse"> <tr> <td> <div onmouseover="this.style.borderColor='#000000'" onmouseout="this.style.borderColor='#9F9F9F'" style="overflow: hidden; width: 150px; height: 95px;border: 1px solid #9F9F9F"> <a title="CBWS.com" href="http://www.cbws.com" onmouseup="setTimeout('NoMove=null',100);" onclick="if(NoMove){return false;}"> <img border="0" src="http://www.iwebtool2.com/img/?r=http://www.homepagestartup.com/&domain=http://www.cbws.com" style="position:relative;left:0px;top:0px;float:center;" width="150"></a></div> </td> </tr> <tr> <td align="center"> <font style="font-size:8pt;font-weight:bold;padding-top:2px;display:block;color:#000000;">CBWS.com</font></td> </tr> </table> </td> </tr> </table></div></div></td><td><div style="width:171px; height:128px;" id="wsm10"><div id="ws10"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 10</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm11"><div id="ws11"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 11</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td><td><div style="width:171px; height:128px;" id="wsm12"><div id="ws12"><table border="0" style="border-collapse: collapse" width="171" height="128" cellpadding="0"> <tr> <td> <table width="100%" height="100%" cellspacing="0" cellpadding="0"><tr><td class="rb" align="center"> <font size="7"> 12</font><font size="4"><br> </font><b><font size="2">No Website Set</font></b></td></tr></table> </td></tr></table></div></div></td></tr><tr></tr></table></div></td> </tr> <tr> <td align="center"> <table border="0" style="border-collapse: collapse" width="98%" id="table1"> <tr> <td> <b><font color="#484848" size="2">Welcome to 00000-NRT-RealEstate's Homepage</font></b></td> <td align="right"> <span style="font-size: 8pt"> <a style="color:#0000FF; font-family:arial" href="what_is_homepagestartup.html"> What is this?</a> | <a href="/" style="color:#0000FF; font-family:arial">Create Your Own Homepage</a></span></td> </tr> </table> </td> </tr> </table> </div> <div id="sbox" style="display:none;position:absolute;top:0px;left:0px;"> <table border="1" style="border-collapse: collapse" bordercolor="#A5ACB8" cellpadding="0" cellspacing="0"> <tr> <td bgcolor="#48505B" height="20" width="130"><font color="#FFFFFF"><span style="font-size: 8pt; font-weight: 700">&nbsp;Change My Search To:</span></font></td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('a');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/a.gif"></td> <td><font style="font-size: 8pt">Google Search</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('b');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/b.gif"></td> <td><font style="font-size: 8pt">MSN Search</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('c');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/c.gif"></td> <td><font style="font-size: 8pt">Yahoo! Search</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('d');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/d.gif"></td> <td><span style="font-size: 8pt">Ask.com Search</span></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('e');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/e.gif"></td> <td><span style="font-size: 8pt">Wikipedia English</span></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('f');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/c.gif"></td> <td><font style="font-size: 8pt">Yahoo! Answers</font></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('g');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/g.gif"></td> <td><span style="font-size: 8pt">Answers.com</span></td> </tr> </table> </td> </tr> <tr> <td bgcolor="#ffffff" onclick="chs_clk('h');" onmouseout="this.style.backgroundColor='#ffffff'" onmouseover="this.style.backgroundColor='#FFFF95';this.style.cursor='pointer';"><table cellpadding="2"> <tr> <td><img border="0" src="/img/h.gif"></td> <td><span style="font-size: 8pt">YouTube Videos</span></td> </tr> </table> </td> </tr> </table></div> </div> </body> </html>"""
# print (create_warc_record(
#     docno='clueweb09-en0000-00-00000',
#     timstamp="2009-01-13 18:05:10",
#     url = 'http://00000-nrt-realestate.homepagestartup.com/',
#     html=html,
#     warc_info_id ='993d3969-9643-4934-b1c6-68d4dbe55b83',
#     warc_date='2009-03-65T08:43:19-0800'))

# print (create_warc_head(warc_date="2009-03-65T08:43:19-0800",
#                         warc_info_id='993d3969-9643-4934-b1c6-68d4dbe55b83'))


